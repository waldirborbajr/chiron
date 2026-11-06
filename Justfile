# ============================================================
# Qwen API — Build Recipes
# ============================================================
#
# Uso:
#   just                  # lista todas as receitas
#   just build-termux     # binário Rust para Termux (Android ARM64)
#   just build-raspi      # binário Rust para Raspberry Pi (Linux ARM64)
#   just build-llama      # llama.cpp cross-compilado para Raspi
#   just build-all        # todos os builds acima
#
# Variáveis configuráveis via env:
#   ANDROID_NDK_HOME  — caminho do Android NDK (obrigatório p/ Termux)
#   API_LEVEL         — API level Android (default: 21)
#   RASPI_TRIPLE      — triple do Pi (default: aarch64-unknown-linux-gnu)
#   LLAMA_REPO        — repositório do llama.cpp

# --- Configuração -------------------------------------------------

android_target   := "aarch64-linux-android"
raspi_target     := "aarch64-unknown-linux-gnu"
api_level        := env_var_or_default("API_LEVEL", "21")
android_ndk      := env_var_or_default("ANDROID_NDK_HOME", "")
llama_repo       := env_var_or_default("LLAMA_REPO", "https://github.com/ggerganov/llama.cpp")
llama_src_dir    := "vendor/llama.cpp"
llama_build_dir  := "vendor/llama.cpp/build-arm"
bin_name         := "qwen-api"

# --- Default: listar receitas -------------------------------------

default:
    @just --list

# ============================================================
# 1. Build para Termux (Android ARM64)
# ============================================================

# Compila o binário Rust para rodar no Termux
build-termux:
    @if [ -z "{{android_ndk}}" ]; then \
        echo "ERRO: ANDROID_NDK_HOME não definido."; \
        echo "  export ANDROID_NDK_HOME=/caminho/para/android-ndk"; \
        exit 1; \
    fi
    @echo "==> Compilando para Termux ({{android_target}}, API {{api_level}})"
    @mkdir -p .cargo
    @printf '[target.{{android_target}}]\nlinker = "{{android_ndk}}/toolchains/llvm/prebuilt/linux-x86_64/bin/{{android_target}}{{api_level}}-clang"\n' > .cargo/config.toml
    cargo build --release --target {{android_target}}
    @echo "==> Binário: target/{{android_target}}/release/{{bin_name}}"

# Empacota o binário Termux em um tarball para transferir
package-termux: build-termux
    @echo "==> Empacotando para Termux"
    @mkdir -p dist
    tar -czf dist/{{bin_name}}-termux.tar.gz \
        -C target/{{android_target}}/release {{bin_name}}
    @echo "==> Pacote: dist/{{bin_name}}-termux.tar.gz"

# ============================================================
# 2. Build para Raspberry Pi (Linux ARM64)
# ============================================================

# Compila o binário Rust para rodar no Raspberry Pi
build-raspi:
    @echo "==> Compilando para Raspberry Pi ({{raspi_target}})"
    @mkdir -p .cargo
    @printf '[target.{{raspi_target}}]\nlinker = "aarch64-linux-gnu-gcc"\n' > .cargo/config.toml
    cargo build --release --target {{raspi_target}}
    @echo "==> Binário: target/{{raspi_target}}/release/{{bin_name}}"

# Empacota o binário Raspi em um tarball
package-raspi: build-raspi
    @echo "==> Empacotando para Raspberry Pi"
    @mkdir -p dist
    tar -czf dist/{{bin_name}}-raspi.tar.gz \
        -C target/{{raspi_target}}/release {{bin_name}}
    @echo "==> Pacote: dist/{{bin_name}}-raspi.tar.gz"

# ============================================================
# 3. Build do llama.cpp cross-compilado para Raspi
# ============================================================

# Clona (se necessário) e compila o llama.cpp para ARM64 Linux
build-llama:
    @echo "==> Preparando llama.cpp em {{llama_src_dir}}"
    @if [ ! -d "{{llama_src_dir}}/.git" ]; then \
        echo "==> Clonando {{llama_repo}}"; \
        mkdir -p vendor; \
        git clone --depth 1 {{llama_repo}} {{llama_src_dir}}; \
    else \
        echo "==> Repositório já existe, atualizando"; \
        git -C {{llama_src_dir}} pull --ff-only; \
    fi
    @echo "==> Configurando CMake para cross-compile ({{raspi_target}})"
    cmake -B {{llama_build_dir}} -S {{llama_src_dir}} -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
    @echo "==> Compilando llama.cpp"
    ninja -C {{llama_build_dir}} -j$(nproc)
    @echo "==> Binários:"
    @ls -lh {{llama_build_dir}}/bin/llama-server {{llama_build_dir}}/bin/llama-cli

# Empacota os binários do llama.cpp para transferir ao Raspi
package-llama: build-llama
    @echo "==> Empacotando llama.cpp para Raspberry Pi"
    @mkdir -p dist/llama-raspi
    cp {{llama_build_dir}}/bin/llama-server dist/llama-raspi/
    cp {{llama_build_dir}}/bin/llama-cli dist/llama-raspi/ 2>/dev/null || true
    tar -czf dist/llama-raspi.tar.gz -C dist llama-raspi
    @echo "==> Pacote: dist/llama-raspi.tar.gz"

# ============================================================
# 4. Builds agregados
# ============================================================

# Compila tudo que é necessário para Raspi (API + llama.cpp)
build-all-raspi: build-raspi build-llama
    @echo "==> Tudo pronto para transferir ao Raspi"
    @echo "  - target/{{raspi_target}}/release/{{bin_name}}"
    @echo "  - {{llama_build_dir}}/bin/llama-server"

# Compila tudo (Termux + Raspi + llama)
build-all: build-termux build-raspi build-llama
    @echo "==> Todos os builds concluídos"

# ============================================================
# 5. Verificações e utilitários
# ============================================================

# Verifica se os cross-compilers estão instalados
check-toolchain:
    @echo "==> Verificando toolchain"
    @command -v rustc       >/dev/null || (echo "FALTA: rustc"; exit 1)
    @command -v cargo       >/dev/null || (echo "FALTA: cargo"; exit 1)
    @command -v cmake       >/dev/null || (echo "FALTA: cmake"; exit 1)
    @command -v ninja       >/dev/null || (echo "FALTA: ninja"; exit 1)
    @command -v aarch64-linux-gnu-gcc >/dev/null || (echo "FALTA: aarch64-linux-gnu-gcc (apt install gcc-aarch64-linux-gnu)"; exit 1)
    @echo "==> Rust targets instalados:"
    @rustup target list --installed | grep -E "aarch64" || echo "  (nenhum target aarch64 instalado)"
    @if [ -n "{{android_ndk}}" ]; then \
        echo "==> Android NDK: {{android_ndk}}"; \
    else \
        echo "==> Android NDK: não definido (Termux build indisponível)"; \
    fi

# Limpa artefatos de build
clean:
    cargo clean
    rm -rf {{llama_build_dir}}
    rm -rf dist
    @echo "==> Limpo"

# Verifica os binários gerados (via file)
inspect:
    @echo "==> Inspecionando binários"
    @for f in \
        target/{{android_target}}/release/{{bin_name}} \
        target/{{raspi_target}}/release/{{bin_name}} \
        {{llama_build_dir}}/bin/llama-server; do \
        if [ -f "$$f" ]; then \
            echo "--- $$f"; \
            file "$$f"; \
        fi; \
    done