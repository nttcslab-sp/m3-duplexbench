#!/usr/bin/env bash
#SBATCH --partition gpu-a6000-kishin,gpu-6000ada,gpu-6000ada-kishin,gpu-a100-kishin,gpu-a100,gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --job-name=m3db
#SBATCH --output=./slurm-log/output_%A_%a.out
#SBATCH --error=./slurm-log/error_%A_%a.err

# https://github.com/de9uch1/argparser
function add_args() {
	argparser setup $0 "Bash script for evaluation pipeline."
	argparser add EXP_DIR -l exp_dir --default outputs/exp/example
	argparser add DATASET -l dataset --choices en_task_topiocqa en_chat_candor ja_task_topiocqa_ja ja_chat_magicdata en_task_topiocqa_bargein en_chat_candor_bargein ja_task_topiocqa_ja_bargein ja_chat_magicdata_bargein en_task_topiocqa_bargein_arg en_chat_candor_bargein_arg ja_task_topiocqa_ja_bargein_arg ja_chat_magicdata_bargein_arg
	argparser add SPLIT -l split --choices test test_small test_other
	# argparser add LANG -l lang --choices en ja
	# argparser add DOMAIN -l domain --choices chat task
	# argparser add DATA -l data --choices topiocqa candor topiocqa_ja magicdata topiocqa_bargein candor_bargein topiocqa_ja_bargein magicdata_bargein
	argparser add MODEL -l model --choices dummy moshi personaplex jmoshi llmjpmoshi freeze_omni moshi_rl personaplex_rl bayling_duplex duplexcascade
	argparser add prepend_silence -l prepend_silence --type bool --default false
	argparser add use_context -l use_context --type bool --default false
	argparser add teacher_forcing -l teacher_forcing --type bool --default false
	argparser add stage -l stage --type int --default 1
	argparser add stop_stage -l stop_stage --type int --default 2
	argparser add dryrun -l dryrun --type bool --default false
	# default url: https://129.60.195.124:8081 (freeze-omni), wss://127.0.0.1:31606 (duplexcascade)
	argparser add server_url -l server_url --type str --default https://129.60.195.124:8081
	argparser add OPENAI_API_KEY -l openai_api_key --type str
	argparser add BAYLING_DUPLEX_PATH -l bayling_duplex_path --type str --default tools/BayLing-Duplex
}
eval $(add_args | argparser parse "$@")

echo "==== Slurm Settings ===="
echo "---- Sl urm Environment Variables ----"
cat <<ETX
hostnme=$(hostname)
JOB_ID="${SLURM_JOB_ID}"
JOB_NAME="${SLURM_JOB_NAME}"
PARTITION_NAME="${SLURM_JOB_PARTITION}"
NODE_LIST="${SLURM_JOB_NODELIST}"
NTASKS="${SLURM_NTASKS}"
CORES_PER_NODE="${SLURM_JOB_CPUS_PER_NODE}"
NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE}"
ETX
echo "==== Slurm Settings ===="

export jmoshi_model=models/hf_hub/llm-jp/llm-jp-moshi-v1/model.safetensors
export jmoshi_config=models/hf_hub/llm-jp/llm-jp-moshi-v1/moshi_lm_kwargs.json

check_variable() {
	local value="$1"
	shift
	local valid_values=("$@")
	for v in "${valid_values[@]}"; do
		if [[ "$value" == "$v" ]]; then
			return 0
		fi
	done
	echo "Error: invalid value: $value" >&2
	echo "Valid values: ${valid_values[*]}" >&2
	exit 1
}

# Variables
IFS=_ read -r LANG DOMAIN DATA <<<"$DATASET"
AUDIO_DIR=data/${LANG}/${DOMAIN}/${DATA}/audio
EVENT_DATA=data/${LANG}/${DOMAIN}/${DATA}/${SPLIT}.jsonl
ALIGNMENT_DIR=data/${LANG}/${DOMAIN}/${DATA}/txt_align
export PYTHONPATH=${PWD}/tools/Freeze-Omni

# exp_name
if [[ $use_context == false ]]; then
	exp_name="${MODEL},use_context=f"
elif [[ $teacher_forcing == false ]]; then
	exp_name="${MODEL},use_context=t,teacher-forcing=f"
else
	exp_name="${MODEL},use_context=t,teacher-forcing=t"
fi
# prepend 5 sec silence for moshi
if [[ $prepend_silence == true ]]; then
	if [[ $MODEL == "moshi" || $MODEL == "moshi_rl" ]]; then
		if [[ $use_context == false || $teacher_forcing == false ]]; then
			exp_name=${exp_name},prepend5sec
		fi
	fi
fi
# channels
if [[ $DOMAIN == "chat" ]]; then
	channels="0,1"
elif [[ $DOMAIN == "task" ]]; then
	channels="1"
fi
# out_dir
out_dir="${EXP_DIR}/${LANG}_${DOMAIN}_${DATA}_${SPLIT}/${exp_name}"
echo out_dir="${out_dir}"
mkdir -p "$out_dir"

[[ "$dryrun" == true ]] && echo "Dry run."

if [[ $stage -le 1 && 1 -le $stop_stage ]]; then
	echo "Stage 1: Inference"
	python ~/Scripts/proxy_py3.py
	case "$MODEL" in
	jmoshi | llmjpmoshi | jmoshiqa | moshi_rl)
		INFER_MODEL=moshi
		;;
	personaplex_rl)
		INFER_MODEL=personaplex
		;;
	*)
		INFER_MODEL=$MODEL
		;;
	esac

	cmd=(
		python -m m3_duplexbench.inference.run_inference
		--channels $channels
		--audio-dir "$AUDIO_DIR"
		--data "$EVENT_DATA"
		--ignore-short-pause
		--output-dir "$out_dir"
		--model "$INFER_MODEL"
		--use-sampling-text
	)

	[[ "$DATA" == *"bargein"* ]] && cmd+=(--ignored-events "TURN_SHIFT,TURN_HOLD,BC")

	[[ $use_context == true ]] && cmd+=(--use-context)

	[[ $teacher_forcing == true ]] && cmd+=(
		--teacher-forcing
		--conditioning-text
		--alignment-dir "$ALIGNMENT_DIR"
	)

	# Model specific args
	if [[ $prepend_silence == true ]]; then
		if [[ $MODEL == "moshi" || $MODEL == "moshi_rl" ]]; then
			if [[ $use_context == false || $teacher_forcing == false ]]; then
				cmd+=(--prepend-silence-sec 5.0)
			fi
		fi
	fi
	[[ "$MODEL" == "jmoshi" ]] && cmd+=(--hf-repo nu-dialogue/j-moshi-ext)
	[[ "$MODEL" == "moshi_rl" ]] && cmd+=(--hf-repo kyutai/moshika-rl-seamless)
	[[ "$MODEL" == "llmjpmoshi" ]] && cmd+=(
		--tokenizer hf://rinna/japanese-gpt2-medium/spiece.model
		--moshi-weight $jmoshi_model
		--config $jmoshi_config
	)
	if [[ "$INFER_MODEL" == "personaplex" ]]; then
		if [[ "$DOMAIN" == "chat" ]]; then
			cmd+=(
				--domain chat
				--voice-prompt NATF2.pt # not used when teacher-forcing is enabled
				--voice-prompt-duration 20
			)
		elif [[ "$DOMAIN" == "task" ]]; then
			cmd+=(
				--domain assistant
				--voice-prompt NATF2.pt # not used when teacher-forcing is enabled
				--voice-prompt-duration 20
			)
		fi
		[[ "$MODEL" == "personaplex" ]] && cmd+=(--hf-repo nvidia/personaplex-7b-v1)
		[[ "$MODEL" == "personaplex_rl" ]] && cmd+=(--hf-repo kyutai/personaplex-rl-seamless)
	fi
	if [[ "$MODEL" == "bayling_duplex" ]]; then
		cmd+=(
			--model_path ${BAYLING_DUPLEX_PATH}/models/bayling_duplex_model
			--speech_tokenizer_path ${BAYLING_DUPLEX_PATH}/models/speech_tokenizer
			--decoder_path ${BAYLING_DUPLEX_PATH}/models/speech_decoder
		)
	fi
	# set server-url for server-client model
	case "$MODEL" in
	freeze_omni | duplexcascade)
		cmd+=(--server-url $server_url)
		;;
	esac

	# Activate conda environments
	if [[ "$INFER_MODEL" == "personaplex" ]]; then
		conda activate personaplex
	elif [[ "$MODEL" == "bayling_duplex" ]]; then
		conda activate bayling-duplex
	else
		conda activate m3-duplexbench
	fi

	echo "${cmd[@]}"
	[[ "$dryrun" != true ]] && "${cmd[@]}" 2>&1 | tee "$out_dir/run_inference.log"
	conda deactivate
fi

if [[ $stage -le 2 && 2 -le $stop_stage ]]; then
	echo "Stage 2: ASR"
	python ~/Scripts/proxy_py3.py
	cmd=(
		python -m m3_duplexbench.asr.run_whisper_asr
		--context-length 1.0
		--root-dir "$out_dir"
		--lang "$LANG"
		--write-audacity-labels
		--metadata metadata.jsonl
	)

	echo "${cmd[@]}"
	[[ "$dryrun" != true ]] && conda run -n m3-duplexbench "${cmd[@]}" 2>&1 | tee "$out_dir/run_asr.log"
fi

if [[ $stage -le 3 && 3 -le $stop_stage ]]; then
	echo "Stage 3: MFA"
	cmd=(
		python -m m3_duplexbench.asr.run_aligner
		--metadata "$out_dir/asr/metadata.jsonl"
		--output-dir "$out_dir/asr"
		--work-dir "$out_dir/asr/workdir"
		--write-audacity-labels
	)
	[[ "$LANG" == "en" ]] && cmd+=(
		--dictionary english_us_arpa
		--acoustic-model english_us_arpa
	)
	[[ "$LANG" == "ja" ]] && cmd+=(
		--dictionary japanese_mfa
		--acoustic-model japanese_mfa
	)

	echo "${cmd[@]}"
	[[ "$dryrun" != true ]] && conda run -n m3-duplexbench "${cmd[@]}" 2>&1 | tee "$out_dir/run_aligner.log"
fi

if [[ $stage -le 4 && 4 -le $stop_stage ]]; then
	echo "Stage 4: Timing Evaluation"
	cmd=(
		python -m m3_duplexbench.evaluation.run_evaluation
		--metadata "$out_dir/asr/metadata_mfa.jsonl"
		--output "$out_dir/evaluation"
		--domain "$DOMAIN"
		--lang "$LANG"
		--eval-category "timing"
	)

	echo "${cmd[@]}"
	[[ "$dryrun" != true ]] && conda run -n m3-duplexbench "${cmd[@]}" 2>&1 | tee "$out_dir/run_evaluation_timing.log"
fi

if [[ $stage -le 5 && 5 -le $stop_stage ]]; then
	echo "Stage 5: Prepare for content Evaluation using LLMAJ"
	python ~/Scripts/proxy_py3.py
	cmd=(
		python -m m3_duplexbench.evaluation.run_evaluation
		--metadata "$out_dir/asr/metadata_mfa.jsonl"
		--output "$out_dir/evaluation"
		--domain "$DOMAIN"
		--lang "$LANG"
		--eval-category "content"
		--eval-mode "prepare"
		--transcript-dir "$ALIGNMENT_DIR"
	)

	echo "${cmd[@]}"
	[[ "$dryrun" != true ]] && conda run -n m3-duplexbench "${cmd[@]}" 2>&1 | tee "$out_dir/run_evaluation_content.log"
fi

if [[ $stage -le 6 && 6 -le $stop_stage ]]; then
	echo "Stage 6: Submit requests to OpenAI API"
	if [ -n "$OPENAI_API_KEY" ]; then
		export OPENAI_API_KEY=$OPENAI_API_KEY
	fi
	mapfile -d '' batch_files < <(
		find "$out_dir/evaluation" -type f -name "batch_requests.jsonl" -print0
	)
	rm ${out_dir}/evaluation/submit_requests.sh
	rm ${out_dir}/evaluation/wait_requests.sh
	for f in "${batch_files[@]}"; do
		echo "python m3_duplexbench/evaluation/utils/openai_batch.py submit --request-jsonl $f --output-dir $(dirname $f)" >>${out_dir}/evaluation/submit_requests.sh
		echo "python m3_duplexbench/evaluation/utils/openai_batch.py wait --batch-info $(dirname $f)/batch_info.json --output-dir $(dirname $f)" >>${out_dir}/evaluation/wait_requests.sh
	done
	conda run -n m3-duplexbench . ${out_dir}/evaluation/submit_requests.sh
fi

if [[ $stage -le 7 && 7 -le $stop_stage ]]; then
	echo "Stage 7: Wait and get results of OpenAI API"
	if [ -n "$OPENAI_API_KEY" ]; then
		export OPENAI_API_KEY=$OPENAI_API_KEY
	fi
	conda run -n m3-duplexbench . ${out_dir}/evaluation/wait_requests.sh
fi

if [[ $stage -le 8 && 8 -le $stop_stage ]]; then
	echo "Stage 8: Content evaluation"
	cmd=(
		python -m m3_duplexbench.evaluation.run_evaluation
		--metadata "$out_dir/asr/metadata_mfa.jsonl"
		--output "$out_dir/evaluation"
		--domain "$DOMAIN"
		--lang "$LANG"
		--eval-category "content"
		--eval-mode "evaluate"
	)

	echo "${cmd[@]}"
	[[ "$dryrun" != true ]] && conda run -n m3-duplexbench "${cmd[@]}" 2>&1 | tee -a "$out_dir/run_evaluation_content.log"
fi
