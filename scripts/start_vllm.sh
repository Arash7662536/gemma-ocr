#!/bin/bash
# Gemma-4 31B on vLLM for gemma-ocr. Run inside tmux; the API key comes from $VLLM_API_KEY (or .env).
#   max_soft_tokens 1120  = Gemma-4's largest image budget (default 280 is too coarse for small Persian digits)
#   --trust-request-mm-kwargs lets clients choose 280 for cheap calls (orientation)
#   --limit-mm-per-prompt image 8 = overview + up to 6 strips
# Stopping: kill the `vllm serve` python process; killing tmux alone leaves it running behind `tee`.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
: "${VLLM_API_KEY:?set VLLM_API_KEY}"
PORT="${PORT:-8001}"
MAX_SEQS="${MAX_SEQS:-16}"
vllm serve google/gemma-4-31B-it --served-model-name google/gemma-4-31B-it \
  --quantization fp8 --kv-cache-dtype fp8 --max-model-len 65536 --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --async-scheduling \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --mm-processor-kwargs '{"max_soft_tokens": 1120}' --trust-request-mm-kwargs \
  --limit-mm-per-prompt '{"image": 8, "video": 0, "audio": 0}' \
  --api-key "$VLLM_API_KEY" --host 127.0.0.1 --port "$PORT" 2>&1 | tee -a vllm.log
