# Terminal 1: iniciar llama-server com o modelo quantizado
./llama-server \
  -m Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 2048 \
  --host 127.0.0.1

# Terminal 2: iniciar sua API Rust
./chiron
# [strategy] Total RAM: 0.95 GB | Available RAM: 0.62 GB
# [strategy] Low RAM detected. Choosing GGUF (via llama-server).
# [init] GGUF strategy selected. Will delegate to llama-server at http://127.0.0.1:8080
# [init] llama-server is reachable.
# [server] Running at http://0.0.0.0:3000 (strategy: Gguf)