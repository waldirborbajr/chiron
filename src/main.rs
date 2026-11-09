use anyhow::{Error, Result, anyhow};
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use sysinfo::System;
use tokenizers::Tokenizer;
use tokio::io::AsyncWriteExt;

const MAX_NEW_TOKENS: usize = 50;
const TEMPERATURE: f64 = 0.3;

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
    model: String,
}

// ============================================================
// Model Selection
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum Strategy {
    Candle, // In-process (Qwen 1.5B)
    Gguf,   // Delegated to llama-server (SmolLM2 or Qwen 0.5B)
}

impl Strategy {
    fn as_str(&self) -> &'static str {
        match self {
            Strategy::Candle => "candle",
            Strategy::Gguf => "gguf",
        }
    }
}

#[derive(Debug, Clone)]
struct ModelConfig {
    strategy: Strategy,
    /// Nome amigável para logs e resposta da API
    name: &'static str,
    /// Repositório HF (usado no download do Candle; nos logs para GGUF)
    hf_repo: &'static str,
    /// RAM total mínima (GiB) para escolher este modelo
    min_ram_gb: f64,
}

const MODEL_SMOLLM2: ModelConfig = ModelConfig {
    strategy: Strategy::Gguf,
    name: "SmolLM2-360M-Instruct-Q4_K_M",
    hf_repo: "bartowski/SmolLM2-360M-Instruct-GGUF",
    min_ram_gb: 0.0,
};

const MODEL_QWEN_05B: ModelConfig = ModelConfig {
    strategy: Strategy::Gguf,
    name: "Qwen2.5-0.5B-Instruct-Q4_K_M",
    hf_repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    min_ram_gb: 1.0,
};

// Os pesos (bf16) são carregados como F32: ~1,54 bi de parâmetros x 4 bytes
// ≈ 6,2 GB. O limite antigo (2,5 GB) estouraria a memória.
const MODEL_QWEN_15B: ModelConfig = ModelConfig {
    strategy: Strategy::Candle,
    name: "Qwen2.5-1.5B-Instruct (safetensors)",
    hf_repo: "Qwen/Qwen2.5-1.5B-Instruct",
    min_ram_gb: 7.0,
};

/// Seleciona o melhor modelo com base na RAM total do dispositivo.
///
/// 1. SmolLM2-360M  (sempre fallback)
/// 2. Qwen2.5-0.5B  (se RAM >= 1.0 GiB, via llama-server)
/// 3. Qwen2.5-1.5B  (se RAM >= 7.0 GiB, via Candle)
fn select_model() -> ModelConfig {
    let mut sys = System::new();
    sys.refresh_memory();

    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let total_ram_gb = sys.total_memory() as f64 / GIB;
    let available_ram_gb = sys.available_memory() as f64 / GIB;

    println!(
        "[select] Total RAM: {total_ram_gb:.2} GiB | Available RAM: {available_ram_gb:.2} GiB"
    );

    if total_ram_gb >= MODEL_QWEN_15B.min_ram_gb {
        println!(
            "[select] Selected: {} (Candle, in-process)",
            MODEL_QWEN_15B.name
        );
        MODEL_QWEN_15B
    } else if total_ram_gb >= MODEL_QWEN_05B.min_ram_gb {
        println!(
            "[select] Selected: {} (GGUF via llama-server)",
            MODEL_QWEN_05B.name
        );
        MODEL_QWEN_05B
    } else {
        println!(
            "[select] Low RAM. Selected: {} (GGUF via llama-server)",
            MODEL_SMOLLM2.name
        );
        MODEL_SMOLLM2
    }
}

// ============================================================
// Candle Inference (in-process, only for Qwen 1.5B)
// ============================================================

struct CandleEngine {
    model: ModelForCausalLM,
    tokenizer: Tokenizer,
    device: Device,
    /// IDs que encerram a geração (<|im_end|>, <|endoftext|>)
    eos_ids: Vec<u32>,
}

impl CandleEngine {
    fn generate(
        &mut self,
        prompt_text: &str,
        max_new_tokens: usize,
        temperature: f64,
    ) -> Result<String> {
        // O KV cache pertence ao modelo: sem limpar, a 2ª requisição
        // reaproveitaria o contexto da 1ª.
        self.model.clear_kv_cache();

        let prompt = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        );
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(299_792_458);
        let mut sampler = LogitsProcessor::new(seed, Some(temperature), None);

        let mut generated: Vec<u32> = Vec::new();
        for index in 0..max_new_tokens {
            // 1º passo: o prompt inteiro. Demais: só o último token, com o
            // offset da posição (o resto já está no KV cache).
            let ctx_len = if index == 0 { tokens.len() } else { 1 };
            let start = tokens.len() - ctx_len;
            let input = Tensor::new(&tokens[start..], &self.device)?.unsqueeze(0)?;

            // forward devolve (batch, 1, vocab)
            let logits = self.model.forward(&input, start)?.squeeze(0)?.squeeze(0)?;
            let next_token = sampler.sample(&logits)?;
            tokens.push(next_token);

            if self.eos_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);
        }

        // Decodifica tudo de uma vez: token a token quebra caracteres
        // multibyte (acentos, emoji) no BPE em nível de byte.
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(Error::msg)?;
        Ok(text.trim().to_string())
    }
}

// ============================================================
// GGUF Inference (delegated to llama-server)
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
    let url = format!("{server_url}/v1/chat/completions");

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
        .map_err(|e| anyhow!("HTTP request to llama-server failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("llama-server returned {status}: {text}"));
    }

    let parsed: LlamaChatResponse = resp
        .json()
        .await
        .map_err(|e| anyhow!("Failed to parse llama-server response: {e}"))?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content.trim().to_string())
        .ok_or_else(|| anyhow!("llama-server returned no choices"))
}

// ============================================================
// Shared Application State
// ============================================================

// Sem Mutex global: o GGUF não precisa de lock (o llama-server cuida da
// própria fila). Só o motor Candle é exclusivo, pois o KV cache é mutável.
struct AppState {
    config: ModelConfig,
    candle: Option<Mutex<CandleEngine>>,
    llama_server_url: String,
    http_client: reqwest::Client,
}

// ============================================================
// API Handler
// ============================================================

fn reply(status: StatusCode, answer: String, state: &AppState) -> impl IntoResponse + use<> {
    (
        status,
        Json(AnswerResponse {
            answer,
            strategy: state.config.strategy.as_str().to_string(),
            model: state.config.name.to_string(),
        }),
    )
}

async fn handle_question(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<QuestionRequest>,
) -> impl IntoResponse {
    let result: Result<String> = match state.config.strategy {
        Strategy::Candle => {
            // Inferência é CPU pesada e síncrona: roda fora do runtime async
            // para não travar as outras conexões.
            let st = state.clone();
            let text = payload.text;
            tokio::task::spawn_blocking(move || {
                let engine = st
                    .candle
                    .as_ref()
                    .ok_or_else(|| anyhow!("Candle model not loaded"))?;
                let mut engine = engine
                    .lock()
                    .map_err(|_| anyhow!("Candle engine lock poisoned"))?;
                engine.generate(&text, MAX_NEW_TOKENS, TEMPERATURE)
            })
            .await
            .map_err(|e| anyhow!("inference task failed: {e}"))
            .and_then(|r| r)
        }
        Strategy::Gguf => {
            generate_with_gguf(
                &state.http_client,
                &state.llama_server_url,
                &payload.text,
                MAX_NEW_TOKENS,
                TEMPERATURE,
            )
            .await
        }
    };

    match result {
        Ok(answer) => reply(StatusCode::OK, answer, &state),
        Err(e) => reply(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Generation error: {e}"),
            &state,
        ),
    }
}

// ============================================================
// HuggingFace download (sem hf-hub: evita native-tls/OpenSSL)
// ============================================================

/// Baixa `file` de `repo` (HuggingFace, branch main) para um cache local
/// em $HOME/.cache/chiron/<repo>/<file>. Se já existir, reutiliza.
async fn hf_download(client: &reqwest::Client, repo: &str, file: &str) -> Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let dest = PathBuf::from(home)
        .join(".cache/chiron")
        .join(repo)
        .join(file);

    if dest.exists() {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    println!("[init] Baixando {url}");
    let mut resp = client.get(&url).send().await?.error_for_status()?;

    // Escreve em partes (chunk) para não carregar o arquivo inteiro na RAM.
    // `.part` é ANEXADO ao nome (with_extension trocaria a extensão).
    let tmp = dest.with_file_name(format!("{file}.part"));
    let mut out = tokio::fs::File::create(&tmp).await?;
    while let Some(chunk) = resp.chunk().await? {
        out.write_all(&chunk).await?;
    }
    out.flush().await?;
    drop(out);
    tokio::fs::rename(&tmp, &dest).await?;
    Ok(dest)
}

// ============================================================
// Main
// ============================================================

async fn load_candle(client: &reqwest::Client, config: &ModelConfig) -> Result<CandleEngine> {
    println!("[init] Loading Candle model (safetensors)...");
    let device = Device::Cpu;

    let tokenizer_path = hf_download(client, config.hf_repo, "tokenizer.json").await?;
    let config_path = hf_download(client, config.hf_repo, "config.json").await?;
    let weights_path = hf_download(client, config.hf_repo, "model.safetensors").await?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;
    let model_config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;

    // SAFETY: o arquivo é mapeado em memória e não pode ser alterado
    // enquanto o modelo estiver em uso.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let model = ModelForCausalLM::new(&model_config, vb)?;

    let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    if eos_ids.is_empty() {
        return Err(anyhow!(
            "tokenizer não conhece <|im_end|> nem <|endoftext|>"
        ));
    }

    println!("[init] Candle model loaded.");
    Ok(CandleEngine {
        model,
        tokenizer,
        device,
        eos_ids,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Seleciona modelo com base na RAM
    let config = select_model();
    let http_client = reqwest::Client::new();

    let llama_server_url =
        std::env::var("LLAMA_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());

    // 2. Carrega recursos conforme a estratégia
    let candle = match config.strategy {
        Strategy::Candle => Some(Mutex::new(load_candle(&http_client, &config).await?)),
        Strategy::Gguf => {
            println!(
                "[init] GGUF strategy. Expecting llama-server at {} with model: {}",
                llama_server_url, config.name
            );

            // Health check
            match http_client
                .get(format!("{llama_server_url}/health"))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    println!("[init] llama-server is reachable.");
                }
                Ok(resp) => {
                    eprintln!(
                        "[init] WARNING: llama-server returned {} at {}",
                        resp.status(),
                        llama_server_url
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[init] WARNING: could not reach llama-server at {llama_server_url}: {e}"
                    );
                }
            }
            None
        }
    };

    // 3. Estado compartilhado
    let shared_state = Arc::new(AppState {
        config: config.clone(),
        candle,
        llama_server_url,
        http_client,
    });

    // 4. Rotas
    let app = Router::new()
        .route("/chiron", post(handle_question))
        .with_state(shared_state);

    // 5. Servidor
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3030").await?;
    println!(
        "[server] Running at http://0.0.0.0:3030 | strategy={:?} | model={}",
        config.strategy, config.name
    );
    axum::serve(listener, app).await?;

    Ok(())
}
