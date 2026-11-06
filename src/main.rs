use anyhow::{Error, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use sysinfo::System;
use tokio::sync::Mutex;
use tokenizers::Tokenizer;

// ============================================================
// API Data Structures
// ============================================================

#[derive(Deserialize)]
struct QuestionRequest {
    text: String,
}

#[derive(Serialize)]
struct AnswerResponse {
    answer: String,
    strategy: String,
}

// ============================================================
// Runtime Strategy Selection
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum Strategy {
    Candle, // Modelo original (safetensors) — ~2GB+ RAM
    Gguf,   // Modelo quantizado via llama-server — ~0.9GB RAM
}

impl Strategy {
    fn as_str(&self) -> &'static str {
        match self {
            Strategy::Candle => "candle",
            Strategy::Gguf => "gguf",
        }
    }
}

fn choose_strategy() -> Strategy {
    let mut sys = System::new_all();
    sys.refresh_all();

    let total_ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let available_ram_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

    println!(
        "[strategy] Total RAM: {:.2} GB | Available RAM: {:.2} GB",
        total_ram_gb, available_ram_gb
    );

    if total_ram_gb < 1.5 {
        println!("[strategy] Low RAM detected. Choosing GGUF (via llama-server).");
        Strategy::Gguf
    } else {
        println!("[strategy] Sufficient RAM. Choosing Candle (original model).");
        Strategy::Candle
    }
}

// ============================================================
// Shared Application State
// ============================================================

struct AppState {
    strategy: Strategy,
    tokenizer: Option<Tokenizer>,
    device: Device,
    candle_model: Option<ModelForCausalLM>,
    llama_server_url: String,
    http_client: reqwest::Client,
}

// ============================================================
// Candle Inference (in-process)
// ============================================================

fn generate_with_candle(
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt_text: &str,
    max_new_tokens: usize,
    temperature: f64,
) -> Result<String> {
    let formatted_prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt_text
    );

    let tokens = tokenizer
        .encode(formatted_prompt, true)
        .map_err(Error::msg)?;
    let mut tokens = tokens.get_ids().to_vec();

    let mut generated_text = String::new();

    for index in 0..max_new_tokens {
        let input = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index)?;

        let logits = logits.squeeze(0)?.squeeze(0)?;
        let logits = (logits / temperature).unwrap_or(logits);

        let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
        tokens.push(next_token);

        if let Ok(text) = tokenizer.decode(&[next_token], true) {
            generated_text.push_str(&text);
        }

        if next_token == 151645 {
            // <|im_end|>
            break;
        }
    }

    Ok(generated_text.replace("<|im_end|>", "").trim().to_string())
}

// ============================================================
// GGUF Inference (delegated to llama-server over HTTP)
// ============================================================

#[derive(Serialize)]
struct LlamaChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct LlamaChatRequest {
    messages: Vec<LlamaChatMessage>,
    max_tokens: usize,
    temperature: f64,
    stream: bool,
}

#[derive(Deserialize)]
struct LlamaChatResponseChoice {
    message: LlamaChatMessageResponse,
}

#[derive(Deserialize)]
struct LlamaChatMessageResponse {
    content: String,
}

#[derive(Deserialize)]
struct LlamaChatResponse {
    choices: Vec<LlamaChatResponseChoice>,
}

async fn generate_with_gguf(
    client: &reqwest::Client,
    server_url: &str,
    prompt_text: &str,
    max_new_tokens: usize,
    temperature: f64,
) -> Result<String> {
    let url = format!("{}/v1/chat/completions", server_url);

    let body = LlamaChatRequest {
        messages: vec![LlamaChatMessage {
            role: "user".to_string(),
            content: prompt_text.to_string(),
        }],
        max_tokens: max_new_tokens,
        temperature,
        stream: false,
    };

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("HTTP request to llama-server failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "llama-server returned {}: {}",
            status,
            text
        ));
    }

    let parsed: LlamaChatResponse = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to parse llama-server response: {}", e))?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content.trim().to_string())
        .ok_or_else(|| anyhow::anyhow!("llama-server returned no choices"))
}

// ============================================================
// API Handler
// ============================================================

async fn handle_question(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(payload): Json<QuestionRequest>,
) -> impl IntoResponse {
    let mut state_guard = state.lock().await;

    let prompt_text = payload.text;
    let max_new_tokens = 50;
    let temperature = 0.3;

    let strategy_str = state_guard.strategy.as_str().to_string();

    let result: Result<String> = match state_guard.strategy {
        Strategy::Candle => {
            let tokenizer = match state_guard.tokenizer.as_ref() {
                Some(t) => t,
                None => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(AnswerResponse {
                            answer: "Tokenizer not loaded".to_string(),
                            strategy: strategy_str,
                        }),
                    )
                }
            };

            if let Some(model) = state_guard.candle_model.as_mut() {
                generate_with_candle(
                    model,
                    tokenizer,
                    &state_guard.device,
                    &prompt_text,
                    max_new_tokens,
                    temperature,
                )
            } else {
                Err(anyhow::anyhow!("Candle model not loaded"))
            }
        }
        Strategy::Gguf => {
            generate_with_gguf(
                &state_guard.http_client,
                &state_guard.llama_server_url,
                &prompt_text,
                max_new_tokens,
                temperature,
            )
            .await
        }
    };

    match result {
        Ok(answer) => (
            StatusCode::OK,
            Json(AnswerResponse {
                answer,
                strategy: strategy_str,
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(AnswerResponse {
                answer: format!("Generation error: {}", e),
                strategy: strategy_str,
            }),
        ),
    }
}

// ============================================================
// Main
// ============================================================

#[tokio::main]
async fn main() -> Result<()> {
    let strategy = choose_strategy();
    let device = Device::Cpu;
    let http_client = reqwest::Client::new();

    // URL do llama-server (configurável via env var)
    let llama_server_url = std::env::var("LLAMA_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());

    // Tokenizer e modelo dependem da estratégia
    let (tokenizer, candle_model) = match strategy {
        Strategy::Candle => {
            println!("[init] Loading tokenizer and Candle model (safetensors)...");
            let api = Api::new()?;
            let repo = api.repo(Repo::new(
                "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
                RepoType::Model,
            ));

            let tokenizer_filename = repo.get("tokenizer.json")?;
            let config_filename = repo.get("config.json")?;
            let weights_filename = repo.get("model.safetensors")?;

            let tokenizer =
                Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
            let config: Config =
                serde_json::from_slice(&std::fs::read(config_filename)?)?;

            let vb = unsafe {
                candle_transformers::models::qwen2::VarBuilder::from_mmaped_safetensors(
                    &[weights_filename],
                    DType::F32,
                    &device,
                )?
            };
            let model = ModelForCausalLM::new(&config, vb)?;

            println!("[init] Candle model loaded.");
            (Some(tokenizer), Some(model))
        }
        Strategy::Gguf => {
            println!(
                "[init] GGUF strategy selected. Will delegate to llama-server at {}",
                llama_server_url
            );

            // Health check do llama-server
            match http_client
                .get(format!("{}/health", llama_server_url))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    println!("[init] llama-server is reachable.");
                }
                Ok(resp) => {
                    eprintln!(
                        "[init] WARNING: llama-server returned status {} at {}. \
                         Make sure it is running before sending requests.",
                        resp.status(),
                        llama_server_url
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[init] WARNING: could not reach llama-server at {}: {}. \
                         Start it with: llama-server -m model.gguf --port 8080",
                        llama_server_url, e
                    );
                }
            }

            (None, None)
        }
    };

    let shared_state = Arc::new(Mutex::new(AppState {
        strategy,
        tokenizer,
        device,
        candle_model,
        llama_server_url,
        http_client,
    }));

    let app = Router::new()
        .route("/chiron", post(handle_question))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3030").await.unwrap();
    println!(
        "[server] Running at http://0.0.0.0:3030 (strategy: {:?})",
        strategy
    );
    axum::serve(listener, app).await.unwrap();

    Ok(())
}