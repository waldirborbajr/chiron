# ============================================================
# Chiron API — Build Recipes (com download de modelos)
# ============================================================
#
# Uso:
#   just                  # lista receitas
#   just build-termux     # binário Rust para Termux
#   just build-raspi      # binário Rust para Raspberry Pi
#   just build-llama      # llama.cpp cross-compilado para Raspi
#   just download-models  # baixa SmolLM2 + Qwen 0.5B GGUF no notebook
#   just package-all-raspi # empacota TUDO para o Raspi (API + llama + modelos)
#
# Variáveis configuráveis via env:
#   ANDROID_NDK_HOME  — caminho do Android NDK
#   API_LEVEL         — API level Android (default: 21)

# --- Configuração -------------------------------------------------

android_target   := "aarch64-linux-android"
raspi_target     := "aarch64-unknown-linux-gnu"
api_level        := env_var_or_default("API_LEVEL", "21")
android_ndk      := env_var_or_default("ANDROID_NDK_HOME", "")
llama_repo       := env_var_or_default("LLAMA_REPO", "https://github.com/ggerganov/llama.cpp")
llama_src_dir    := "vendor/llama.cpp"
llama_build_dir  := "vendor/llama.cpp/build-arm"
bin_name         := "chiron"
models_dir       := "dist/models"

# Modelos GGUF a baixar (leves para Raspi)
smollm2_repo     := "bartowski/SmolLM2-360M-Instruct-GGUF"
smollm2_file     := "SmolLM2-360M-Instruct-Q4_K_M.gguf"
qwen05b_repo     := "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
qwen05b_file     := "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# ============================================================
# Default
# ============================================================

default:
    @just --list

# ============================================================
# 1. Build Termux
# ============================================================

build-termux:
    @if [ -z "{{android_ndk}}" ]; then \
        echo "ERRO: ANDROID_NDK_HOME não definido."; \
        exit 1; \
    fi
    @echo "==> Compilando para Termux ({{android_target}})"
    @mkdir -p .cargo
    @printf '[target.{{android_target}}]\nlinker = "{{android_ndk}}/toolchains/llvm/prebuilt/linux-x86_64/bin/{{android_target}}{{api_level}}-clang"\n' > .cargo/config.toml
    cargo build --release --target {{android_target}}
    @echo "==> Binário: target/{{android_target}}/release/{{bin_name}}"

package-termux: build-termux
    @mkdir -p dist
    tar -czf dist/{{bin_name}}-termux.tar.gz -C target/{{android_target}}/release {{bin_name}}
    @echo "==> Pacote: dist/{{bin_name}}-termux.tar.gz"

# ============================================================
# 2. Build Raspi (Rust)
# ============================================================

build-raspi:
    @echo "==> Compilando para Raspberry Pi ({{raspi_target}})"
    @mkdir -p .cargo
    @printf '[target.{{raspi_target}}]\nlinker = "aarch64-linux-gnu-gcc"\n' > .cargo/config.toml
    cargo build --release --target {{raspi_target}}
    @echo "==> Binário: target/{{raspi_target}}/release/{{bin_name}}"

# ============================================================
# 3. Build llama.cpp para Raspi
# ============================================================

build-llama:
    @if [ ! -d "{{llama_src_dir}}/.git" ]; then \
        echo "==> Clonando llama.cpp"; \
        mkdir -p vendor; \
        git clone --depth 1 {{llama_repo}} {{llama_src_dir}}; \
    else \
        echo "==> Atualizando llama.cpp"; \
        git -C {{llama_src_dir}} pull --ff-only; \
    fi
    @echo "==> Configurando CMake para cross-compile"
    cmake -B {{llama_build_dir}} -S {{llama_src_dir}} -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
    @ninja -C {{llama_build_dir}} -j$(nproc)
    @ls -lh {{llama_build_dir}}/bin/llama-server

# ============================================================
# 4. Download de Modelos GGUF (no notebook)
# ============================================================

# Baixa SmolLM2-360M e Qwen2.5-0.5B (Q4_K_M) para dist/models/
download-models:
    @mkdir -p {{models_dir}}
    @echo "==> Baixando SmolLM2-360M..."
    @curl -L --fail -o {{models_dir}}/{{smollm2_file}} \
        "https://huggingface.co/{{smollm2_repo}}/resolve/main/{{smollm2_file}}"
    @echo "==> Baixando Qwen2.5-0.5B..."
    @curl -L --fail -o {{models_dir}}/{{qwen05b_file}} \
        "https://huggingface.co/{{qwen05b_repo}}/resolve/main/{{qwen05b_file}}"
    @ls -lh {{models_dir}}

# ============================================================
# 5. Pacotes agregados para Raspi
# ============================================================

# Empacota apenas a API Rust
package-raspi: build-raspi
    @mkdir -p dist
    tar -czf dist/{{bin_name}}-raspi.tar.gz -C target/{{raspi_target}}/release {{bin_name}}
    @echo "==> Pacote: dist/{{bin_name}}-raspi.tar.gz"

# Empacota llama-server para Raspi
package-llama: build-llama
    @mkdir -p dist/llama-raspi
    cp {{llama_build_dir}}/bin/llama-server dist/llama-raspi/
    cp {{llama_build_dir}}/bin/llama-cli dist/llama-raspi/ 2>/dev/null || true
    tar -czf dist/llama-raspi.tar.gz -C dist llama-raspi
    @echo "==> Pacote: dist/llama-raspi.tar.gz"

# TUDO para Raspi: API + llama.cpp + modelos GGUF
package-all-raspi: build-raspi build-llama download-models
    @echo "==> Empacotando bundle completo para Raspi"
    @mkdir -p dist/bundle-raspi
    cp target/{{raspi_target}}/release/{{bin_name}} dist/bundle-raspi/
    cp {{llama_build_dir}}/bin/llama-server dist/bundle-raspi/
    cp -r {{models_dir}} dist/bundle-raspi/models/
    tar -czf dist/bundle-raspi.tar.gz -C dist bundle-raspi
    @echo "==> Bundle: dist/bundle-raspi.tar.gz"
    @echo "    Conteúdo:"
    @echo "      - {{bin_name}}"
    @echo "      - llama-server"
    @echo "      - models/{{smollm2_file}}"
    @echo "      - models/{{qwen05b_file}}"

# ============================================================
# 6. Utilitários
# ============================================================

check-toolchain:
    @command -v rustc >/dev/null || (echo "FALTA: rustc"; exit 1)
    @command -v cmake >/dev/null || (echo "FALTA: cmake"; exit 1)
    @command -v ninja >/dev/null || (echo "FALTA: ninja"; exit 1)
    @command -v aarch64-linux-gnu-gcc >/dev/null || (echo "FALTA: aarch64-linux-gnu-gcc"; exit 1)
    @echo "==> Toolchain OK"

clean:
    cargo clean
    rm -rf {{llama_build_dir}} dist
    @echo "==> Limpo"
