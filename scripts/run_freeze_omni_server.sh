#!/bin/bash
cd tools/Freeze-Omni

PYTHONPATH=
export PYTHONPATH=./:$PYTHONPATH

ip=0.0.0.0      # Replace with your server's IP address.
port=8080       # Replace with your server's port.
llm_exec_nums=1 # Recommended to set to 1, requires about 15GB GPU memory per exec. Try setting a value greater than 1 on a better GPU to improve concurrency performance.
max_users=10    # Maximum number of users allowed to connect at the same time. Requires about 2GB GPU memory per max_users, adjust according to GPU memory size.
timeout=180     # Timeout for each user.

model_path=./checkpoints     # Replace with your model path (download from our huggingface repo).
llm_path=./Qwen2-7B-Instruct # Replace with your Qwen2-7B-Instruct model path.

top_p=0.8
top_k=20
temperature=0.8

conda activate freeze-omni
CUDA_VISIBLE_DEVICES=0 python3 bin/server.py \
  --ip $ip \
  --port $port \
  --max_users $max_users \
  --llm_exec_nums $llm_exec_nums \
  --timeout $timeout \
  --model_path $model_path \
  --llm_path $llm_path \
  --top_p ${top_p} \
  --top_k ${top_k} \
  --temperature ${temperature}
