# M3-DuplexBench: A Multi-Turn, Multilingual, Multidomain Benchmark for Full-Duplex Spoken Dialogue Models

M3-DuplexBench evaluates interaction timing and response content in full-duplex
spoken dialogue models under controlled single-turn and multi-turn conditions.
It covers English and Japanese, including casual conversation and multi-turn QA.

## Contents
- [Installation](#installation)
- [Data Preparation (WIP)](#data-preparation-wip)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Benchmark Specification](#benchmark-specification)
- [Licence](#licence)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Installation

Set up the default environment for Moshi-family inference and benchmark
post-processing. Other model families require separate environments or servers.

### Moshi Family (Default Environment)
Run these commands from the repository root:

```bash
git submodule update --init tools/moshi
bash scripts/apply_moshi_patches.sh
cd envs
conda env create -f environment.yml
cd ..
conda activate m3-duplexbench
```


### PersonaPlex

Update and patch the submodule, then create its dedicated environment:

```bash
git submodule update --init tools/personaplex
bash scripts/apply_personaplex_patches.sh
conda create -n personaplex python=3.10 -y
conda activate personaplex
```

Follow the [official README](https://github.com/NVIDIA/personaplex#usage) to
install the patched code in this environment.

### Freeze-Omni (Server Environment)

Update and patch the submodule:

```bash
git submodule update --init tools/Freeze-Omni
bash scripts/apply_freeze_omni_patches.sh
```

Follow the [official README](https://github.com/VITA-MLLM/Freeze-Omni) to set up
the server environment from `tools/Freeze-Omni`. Run the benchmark client in
`m3-duplexbench`.

### DuplexCascade (Server Environment)

Update and patch the submodule:

```bash
git submodule update --init tools/DuplexCascade
bash scripts/apply_duplexcascade_patches.sh
```

Follow the [official README](https://github.com/sbintuitions/DuplexCascade) to
set up the server environment from `tools/DuplexCascade`. Run the benchmark
client in `m3-duplexbench`.

### BayLing-Duplex

Update the submodule:

```bash
git submodule update --init tools/BayLing-Duplex
```

Follow the [official README](https://github.com/BayLing-Models/BayLing-Duplex) to
install the code in a conda environment named `bayling-duplex`.

## Data Preparation (WIP)

Data download and preprocessing instructions are in progress. Arrange each
dataset under `data/{lang}/{domain}/{dataset}/` as follows:

- `audio/{id}.wav`: two-channel audio, with one speaker per channel.
- `{split}.jsonl`: dialogue IDs and event annotations (see [JSON Data Format](#json-data-format)).
- `txt_align/{id}.txt`: reference word alignments for text-conditioned teacher
  forcing and for content evaluation using reference transcripts.

Inference examples assume these files are already available.

## Evaluation

Run all stages with `scripts/pipe.sh`, or execute them individually using the
[step-by-step workflow](#step-by-step-workflow). Run commands from the repository root.

```bash
# Example: Moshi on Candor with teacher-forced context.
bash scripts/pipe.sh \
  --exp_dir outputs/exp/example \
  --dataset en_chat_candor --split test_small --model moshi \
  --use_context true --teacher_forcing true --stage 1 --stop_stage 4

# Show all available options and defaults.
bash scripts/pipe.sh --help
```

| Stage | Operation |
|---|---|
| 1 | Model inference |
| 2 | Transcription with faster-whisper |
| 3 | Word alignment with Montreal Forced Aligner (MFA) |
| 4 | Timing evaluation |
| 5 | Preparation of content-evaluation requests |
| 6 | Submission to the OpenAI Batch API |
| 7 | Waiting for completion and downloading results |
| 8 | Content evaluation |

`--stage` and `--stop_stage` select an inclusive range of stages.

### Step-by-Step Workflow

Keep the variables set in Stage 1 for the following commands. Run the workflow
in the `m3-duplexbench` environment.

#### Stage 1. Inference
Here is an example of J-Moshi inference with teacher forcing on Japanese
multi-turn QA data.

```bash
LANG=ja
DOMAIN=task
DATA=topiocqa_ja
SPLIT=test
out_dir="outputs/${LANG}_${DOMAIN}_${DATA}_${SPLIT}/jmoshi_context_tf"
python -m m3_duplexbench.inference.run_inference \
  --model moshi \
  --hf-repo nu-dialogue/j-moshi-ext \
  --audio-dir "data/${LANG}/${DOMAIN}/${DATA}/audio" \
  --data "data/${LANG}/${DOMAIN}/${DATA}/${SPLIT}.jsonl" \
  --output-dir "$out_dir" \
  --channels 1 \
  --ignore-short-pause \
  --use-sampling-text \
  --use-context \
  --teacher-forcing \
  --conditioning-text \
  --alignment-dir "data/${LANG}/${DOMAIN}/${DATA}/txt_align"
```
`--ignore-short-pause` excludes `SHORTPAUSE` events; `TURN_HOLD` events remain
available for pause-handling evaluation. In task-oriented data, `--channels 1`
selects the annotated channel to evaluate; chat data uses `--channels 0,1`.

See [Inference](#inference) for model-specific settings.

#### Stage 2. Transcription with faster-whisper
```bash
python -m m3_duplexbench.asr.run_whisper_asr \
  --context-length 1.0 \
  --root-dir "$out_dir" \
  --lang "$LANG" \
  --write-audacity-labels \
  --metadata metadata.jsonl
```
`--context-length 1.0` prepends up to one second of preceding audio to the ASR
input when a context audio file is available.

#### Stage 3. Word Alignment with MFA
```bash
python -m m3_duplexbench.asr.run_aligner \
  --metadata "${out_dir}/asr/metadata.jsonl" \
  --output-dir "${out_dir}/asr" \
  --work-dir "${out_dir}/asr/workdir" \
  --write-audacity-labels \
  --dictionary japanese_mfa \
  --acoustic-model japanese_mfa
```
For English, set both `--dictionary` and `--acoustic-model` to
`english_us_arpa`. For Japanese, use `japanese_mfa` as shown above. MFA must be
installed, and the requested dictionary and acoustic model must be available.

#### Stage 4. Timing Evaluation
```bash
python -m m3_duplexbench.evaluation.run_evaluation \
  --metadata "${out_dir}/asr/metadata_mfa.jsonl" \
  --output "${out_dir}/evaluation" \
  --domain "$DOMAIN" \
  --lang "$LANG" \
  --eval-category timing
```

#### Stages 5–8. Content Evaluation
Content evaluation requires access to the OpenAI Batch API and incurs API charges.
Evaluators use `gpt-5-nano` by default. First, prepare Batch API requests:

```bash
python -m m3_duplexbench.evaluation.run_evaluation \
  --metadata "${out_dir}/asr/metadata_mfa.jsonl" \
  --output "${out_dir}/evaluation" \
  --domain "$DOMAIN" \
  --lang "$LANG" \
  --eval-category content \
  --eval-mode prepare \
  --transcript-dir "data/${LANG}/${DOMAIN}/${DATA}/txt_align"
```

Then, export your OpenAI API key and submit the batches. Set the key locally;
do not commit it to the repository. Batch completion is asynchronous.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

# Find prepared request files (requires Bash).
mapfile -d '' batch_files < <(
  find "$out_dir/evaluation" -type f -name "batch_requests.jsonl" -print0
)
for f in "${batch_files[@]}"; do
  python m3_duplexbench/evaluation/utils/openai_batch.py submit \
    --request-jsonl "$f" --output-dir "$(dirname "$f")"
done

# Wait for completion and download results.
for f in "${batch_files[@]}"; do
  python m3_duplexbench/evaluation/utils/openai_batch.py wait \
    --batch-info "$(dirname "$f")/batch_info.json" \
    --output-dir "$(dirname "$f")"
done
```

Finally, calculate the content metrics:

```bash
python -m m3_duplexbench.evaluation.run_evaluation \
  --metadata "${out_dir}/asr/metadata_mfa.jsonl" \
  --output "${out_dir}/evaluation" \
  --domain "$DOMAIN" \
  --lang "$LANG" \
  --eval-category content \
  --eval-mode evaluate
```

## Inference

### Context Modes
| Configuration | Python CLI options |
|---|---|
| Single-turn (no dialogue history) | Omit `--use-context`, `--teacher-forcing`, `--conditioning-text`, and `--alignment-dir` |
| Multi-turn with user-audio context | `--use-context` |
| Multi-turn with teacher-forced audio and text context | `--use-context --teacher-forcing --conditioning-text --alignment-dir PATH` |

Single-turn evaluation follows the no-context setup used by
[Full-Duplex-Bench v1.0](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v1_v1.5).
Teacher forcing conditions generation on reference system-side audio. Adding
`--conditioning-text` also uses reference word alignments. See
[Supported Models](#supported-models) for compatibility.

Use `--context-max-len SECONDS` to limit the amount of dialogue history.
For Moshi inference without teacher forcing, `--prepend-silence-sec 5.0` prepends
five seconds of silence. In `scripts/pipe.sh`, the corresponding option is
`--prepend_silence true`, which applies to `moshi` and `moshi_rl`.

### Local Checkpoints
To use local checkpoints, specify the tokenizer, model weights, and model
configuration explicitly. These options override the corresponding files from
the default Hugging Face repository.

```bash
python -m m3_duplexbench.inference.run_inference \
  --model moshi \
  --tokenizer "$PATH_TO_TOKENIZER" \
  --moshi-weight "$PATH_TO_CHECKPOINT" \
  --config "$PATH_TO_CONFIG" \
  --audio-dir "data/${LANG}/${DOMAIN}/${DATA}/audio" \
  --data "data/${LANG}/${DOMAIN}/${DATA}/${SPLIT}.jsonl" \
  --output-dir "$out_dir"
```

### Model-Specific Options

Replace `--model moshi` and add the model-specific options below.
Python model names differ from some pipeline aliases:

| Model | Pipeline `--model` | Python CLI options |
|---|---|---|
| Moshi (Moshiko) | `moshi` | `--model moshi` |
| J-Moshi | `jmoshi` | `--model moshi --hf-repo nu-dialogue/j-moshi-ext` |
| Moshi-RL | `moshi_rl` | `--model moshi --hf-repo kyutai/moshika-rl-seamless` |
| LLM-jp-Moshi | `llmjpmoshi` | `--model moshi` with the tokenizer, weights, and config below |
| PersonaPlex | `personaplex` | `--model personaplex --hf-repo nvidia/personaplex-7b-v1` |
| PersonaPlex-RL | `personaplex_rl` | `--model personaplex --hf-repo kyutai/personaplex-rl-seamless` |
| Freeze-Omni | `freeze_omni` | `--model freeze_omni --server-url URL` |
| DuplexCascade | `duplexcascade` | `--model duplexcascade --server-url URL` |
| BayLing-Duplex | `bayling_duplex` | `--model bayling_duplex` with the checkpoint paths below |
| Dummy (testing only) | `dummy` | `--model dummy` |

**Moshi family:** Activate `m3-duplexbench`. Optionally add
`--use-sampling-text`. For LLM-jp-Moshi, download
`model.safetensors` and `moshi_lm_kwargs.json` from
[llm-jp/llm-jp-moshi-v1](https://huggingface.co/llm-jp/llm-jp-moshi-v1), then add:

```text
--tokenizer hf://rinna/japanese-gpt2-medium/spiece.model
--moshi-weight /path/to/model.safetensors
--config /path/to/moshi_lm_kwargs.json
```

**PersonaPlex / PersonaPlex-RL:** Activate `personaplex` and add:

```text
--domain chat
--voice-prompt NATF2.pt
--voice-prompt-duration 20
```

Use `--domain assistant` for multi-turn QA. The voice prompt is not used when
teacher forcing is enabled. Optionally add `--cpu-offload` if GPU memory is
insufficient (requires `accelerate`).

**Freeze-Omni:** Start the server in its dedicated environment following the
[official demo instructions](https://github.com/VITA-MLLM/Freeze-Omni).
Run the benchmark client in `m3-duplexbench` with
`--model freeze_omni --server-url https://HOST:PORT`, using the server's actual
address and protocol.

**DuplexCascade:** Start the ASR, TTS, and main server components following the
[official demo instructions](https://github.com/sbintuitions/DuplexCascade).
Run the benchmark client in `m3-duplexbench` with
`--model duplexcascade --server-url wss://HOST:31606` (use `ws://` without TLS).

**BayLing-Duplex:** Activate `bayling-duplex`, download the checkpoints following
the [official README](https://github.com/BayLing-Models/BayLing-Duplex), and add:

```text
--model_path tools/BayLing-Duplex/models/bayling_duplex_model
--speech_tokenizer_path tools/BayLing-Duplex/models/speech_tokenizer
--decoder_path tools/BayLing-Duplex/models/speech_decoder
```

### Adding a Model
Implement a wrapper in `m3_duplexbench/inference/models/` using
[BaseInferenceModel](m3_duplexbench/inference/models/base.py) as the interface.
Provide `name`, `sample_rate`, `context_mode`, and `generate()`, which returns
user audio, system audio, and optional generated context audio. Register the
wrapper in `create_model()` and the `--model` choices in
[run_inference.py](m3_duplexbench/inference/run_inference.py). To expose it through
the shell pipeline, also update the model choices and model-specific arguments
in [scripts/pipe.sh](scripts/pipe.sh).

## Benchmark Specification
### Structure
```text
.
├── m3_duplexbench/
│   ├── inference/        # Event-based inference and model wrappers
│   ├── asr/              # Transcription and forced alignment
│   └── evaluation/       # Timing and content evaluators
├── scripts/              # Pipeline entry points
├── envs/                 # Environment definitions
├── data/                 # {lang}/{domain}/{dataset}
├── outputs/              # Generated audio, transcripts, and evaluation results
└── tools/                # External repositories and submodules
```

### Datasets

| Pipeline dataset name | Data directory | Domain |
|---|---|---|
| `en_task_topiocqa` | `data/en/task/topiocqa/` | Multi-turn QA |
| `en_chat_candor` | `data/en/chat/candor/` | Casual conversation |
| `ja_task_topiocqa_ja` | `data/ja/task/topiocqa_ja/` | Multi-turn QA |
| `ja_chat_magicdata` | `data/ja/chat/magicdata/` | Casual conversation |

The pipeline also accepts `_bargein`/`_bargein_arg` variants of these dataset
names, with separate dataset directories.
Split names are `test`, `test_small`, and `test_other`.

```text
data/{lang}/{domain}/{dataset}/
├── test.jsonl
├── test_small.jsonl       # If available
├── test_other.jsonl       # If available
├── audio/
│   └── {id}.wav
└── txt_align/
    └── {id}.txt
```

#### JSON Data Format

Each line in a split file is a JSON object with a dialogue `id` and an `events`
mapping. The ID must match the audio filename without the `.wav` extension.
Channel keys are `"0"` and `"1"`, and event times are `[start, end]` in seconds.
For readability, one record is shown below across multiple lines; in a JSONL
file, write each complete record on a single line.

```json
{
  "id": "example_dialogue",
  "events": {
    "0": [
      {"event": "TURN", "time": [0.16, 1.04]},
      {"event": "TURN_HOLD", "time": [1.04, 1.54]},
      {"event": "TURN", "time": [1.54, 2.50]}
    ],
    "1": [
      {"event": "TURN_SHIFT", "time": [2.60, 2.61]},
      {"event": "TURN", "time": [2.61, 4.00]}
    ]
  }
}
```

Recognized event types are `TURN`, `TURN_SHIFT`, `TURN_HOLD`, `SHORTPAUSE`, `BC`,
and `BARGE_IN`. `TURN` annotations mark inter-pausal units; the remaining types
identify evaluation events.

### Supported models
| Model | Single-turn (no context) | Multi-turn (generated context) | Multi-turn (reference context) |
| ---- | :----: | :----: | :----: |
| Moshi and Moshi-based variants (offline) | ✔︎ | ✔︎ | ✔︎ |
| PersonaPlex and PersonaPlex-RL (offline) | ✔︎ | ✔︎ | ✔︎ |
| BayLing-Duplex (offline) | ✔︎ | ✔︎ | — |
| Freeze-Omni (server-client) | ✔︎ | ✔︎ | — |
| DuplexCascade (server-client) | ✔︎ | ✔︎ | — |

Inference aliases and model-specific options are listed in
[Model-Specific Options](#model-specific-options). `dummy` is available for testing.

### Evaluation metrics
| Category | Dimension | Metric |
|---|---|---|
| Timing | Smooth turn taking | TOR↑, Latency↓ |
| Timing | Pause handling | TOR↓, Latency↑ |
| Timing | User backchanneling | Stop latency↑ |
| Timing | User barge-in | Stop latency↓ |
| Content | Response relevance | LLM-as-a-judge |
| Content | Contextual consistency | LLM-as-a-judge |
| Content | QA accuracy (task domain only) | LLM-as-a-judge |

The active dimensions are defined in [evaluation/registry.py](m3_duplexbench/evaluation/registry.py).
## Licence
See [LICENCE](LICENCE.pdf) for details.

## Citation

If you find our benchmark useful in your research, please consider citing our paper.

```bibtex
@article{fukuda2026m3db,
      title={M3-DuplexBench: A Multi-Turn, Multilingual, Multidomain Benchmark for Full-Duplex Spoken Dialogue Models}, 
      author={Ryo Fukuda and Atsushi Ando and Hiroki Kanagawa and Takatomo Kano and Marc Delcroix and Naohiro Tawara and Yuya Chiba},
      journal={arXiv preprint arXiv:2607.29125},
      year={2026}
}
```

## Acknowledgements
This project builds on
[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench).

