echo "Launching sglang server on port ${SGLANG_PORT}"
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --port $SGLANG_PORT > /dev/null 2>&1 &
