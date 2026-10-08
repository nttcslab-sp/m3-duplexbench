server=$1   # asr, tts or llm
ssl_cert=$2 # ssl certificate directory

conda activate DuplexCascade

if [ "$server" == asr ]; then
	echo "Running ASR server"
	moshi-server worker \
		--addr 127.0.0.1 \
		--port 31607 \
		--config tools/delayed-streams-modeling/configs/config-stt-en_fr-hf.toml
elif [ "$server" == "tts" ]; then
	echo "Running TTS server"
	export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64"
	moshi-server worker \
		--addr 127.0.0.1 \
		--port 31608 \
		--config tools/delayed-streams-modeling/configs/config-tts.toml
elif [ "$server" == "llm" ]; then
	echo "Running LLM server"
	python tools/DuplexCascade/server.py \
		--port 31606 \
		--ssl-certfile ${ssl_cert}/cert.pem \
		--ssl-keyfile ${ssl_cert}/key.pem
else
	echo "No ${server} server found"
fi
