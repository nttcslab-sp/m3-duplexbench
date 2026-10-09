# M3-DuplexBench: A Multi-Turn, Multilingual, Multidomain Benchmark for Full-Duplex Spoken Dialogue Models

M3-DuplexBench evaluates interaction timing and response content in full-duplex
spoken dialogue models under controlled single-turn and multi-turn conditions.
It covers English and Japanese, including casual conversation and multi-turn QA.

## 📊 Evaluation Results
The following are the evaluation results under three different contextual settings.
See [our paper](#citation) for details.

<details>
<summary>Single-turn evaluation</summary>

No preceding dialogue history is provided to the model.
Values are reported as mean ± 95% confidence interval.

<table>
<thead>
<tr>
  <th rowspan="3">Model</th>
  <th colspan="8">Chat (Real data)</th>
  <th colspan="6">Multi-turn QA (Synthetic data)</th>
</tr>
<tr>
  <th colspan="2">Smooth turn-taking</th>
  <th colspan="2">Pause handling</th>
  <th>User backchannel</th>
  <th>User barge-in</th>
  <th colspan="2">Response quality</th>
  <th colspan="2">Smooth turn-taking</th>
  <th>User barge-in</th>
  <th colspan="3">Response quality</th>
</tr>
<tr>
  <th>TOR ↑</th>
  <th>Latency ↓</th>
  <th>TOR ↓</th>
  <th>Latency ↑</th>
  <th>Stop latency ↑</th>
  <th>Stop latency ↓</th>
  <th>Relevance ↑</th>
  <th>Consistency ↑</th>
  <th>TOR ↑</th>
  <th>Latency ↓</th>
  <th>Stop latency ↓</th>
  <th>Relevance ↑</th>
  <th>Consistency ↑</th>
  <th>QA Score ↑</th>
</tr>
</thead>
<tbody>
<tr><th colspan="15">English</th></tr>
<tr><td>DuplexCascade</td><td>72.7±5.5</td><td>4.70±0.42</td><td><strong>0.3</strong>±0.5</td><td><strong>6.91</strong>±0.30</td><td>—</td><td>—</td><td>61.3±2.6</td><td>77.7±3.2</td><td>92.4±2.3</td><td>2.78±0.10</td><td>—</td><td>63.0±2.2</td><td><strong>70.2</strong>±2.4</td><td><strong>19.8</strong>±2.6</td></tr>
<tr><td>Freeze-Omni</td><td>82.7±4.6</td><td>3.06±0.35</td><td>12.5±3.3</td><td>5.55±0.38</td><td>—</td><td>—</td><td>70.4±3.2</td><td>80.2±3.2</td><td>90.5±2.5</td><td>1.71±0.13</td><td>—</td><td>62.6±2.4</td><td>61.4±2.9</td><td>16.7±2.5</td></tr>
<tr><td>BayLing-Duplex</td><td>85.4±4.3</td><td>2.49±0.17</td><td>14.8±3.5</td><td>3.66±0.29</td><td>—</td><td>—</td><td>72.5±3.6</td><td>59.4±4.0</td><td>87.5±2.8</td><td>2.91±0.14</td><td>—</td><td>71.3±2.4</td><td>62.4±2.8</td><td>19.6±2.6</td></tr>
<tr><td>Moshi</td><td>42.3±6.0</td><td>2.14±0.56</td><td>39.8±4.8</td><td>4.57±0.48</td><td>—</td><td>—</td><td>62.3±2.9</td><td>63.9±3.2</td><td>41.8±4.2</td><td>2.46±0.39</td><td>—</td><td>60.6±2.1</td><td>54.1±2.2</td><td>11.1±2.2</td></tr>
<tr><td>Moshi-RL</td><td>95.8±2.5</td><td>2.15±0.32</td><td>35.5±4.7</td><td>1.73±0.23</td><td>—</td><td>—</td><td><strong>78.3</strong>±3.1</td><td>68.3±3.3</td><td>97.0±1.5</td><td>2.44±0.22</td><td>—</td><td>68.4±2.3</td><td>50.2±2.1</td><td>14.6±2.4</td></tr>
<tr><td>PersonaPlex</td><td>94.6±2.8</td><td>0.78±0.14</td><td>32.5±4.6</td><td>3.29±0.35</td><td>—</td><td>—</td><td>70.8±3.4</td><td>72.3±3.3</td><td>97.2±1.4</td><td>0.38±0.03</td><td>—</td><td>76.9±2.4</td><td>56.9±2.9</td><td>15.7±2.3</td></tr>
<tr><td>PersonaPlex-RL</td><td><strong>99.2</strong>±1.1</td><td><strong>0.59</strong>±0.07</td><td>17.0±3.7</td><td>2.40±0.17</td><td>—</td><td>—</td><td>77.1±3.2</td><td><strong>78.7</strong>±3.2</td><td><strong>97.9</strong>±1.2</td><td><strong>0.31</strong>±0.01</td><td>—</td><td><strong>79.9</strong>±2.3</td><td>59.5±2.9</td><td>17.9±2.5</td></tr>
<tr><th colspan="15">Japanese</th></tr>
<tr><td>J-Moshi</td><td>69.9±4.8</td><td><strong>1.65</strong>±0.25</td><td><strong>23.4</strong>±4.6</td><td><strong>5.82</strong>±0.46</td><td>—</td><td>—</td><td><strong>71.2</strong>±3.2</td><td><strong>88.4</strong>±2.4</td><td>90.8±2.5</td><td><strong>1.08</strong>±0.14</td><td>—</td><td><strong>58.0</strong>±2.2</td><td><strong>63.0</strong>±2.5</td><td><strong>11.5</strong>±2.2</td></tr>
<tr><td>LLM-jp-Moshi</td><td><strong>96.3</strong>±2.0</td><td>2.81±0.25</td><td>28.5±4.9</td><td>3.36±0.31</td><td>—</td><td>—</td><td>59.9±2.3</td><td>79.4±2.7</td><td><strong>92.4</strong>±2.3</td><td>2.76±0.20</td><td>—</td><td>52.2±1.8</td><td>61.0±2.4</td><td>10.2±1.9</td></tr>
</tbody>
</table>

</details>

<details>
<summary>Multi-turn evaluation</summary>

Free-running multi-turn evaluation with reference user history and model-generated system history.

<table>
<thead>
<tr>
  <th rowspan="3">Model</th>
  <th colspan="8">Chat (Real data)</th>
  <th colspan="6">Multi-turn QA (Synthetic data)</th>
</tr>
<tr>
  <th colspan="2">Smooth turn-taking</th>
  <th colspan="2">Pause handling</th>
  <th>User backchannel</th>
  <th>User barge-in</th>
  <th colspan="2">Response quality</th>
  <th colspan="2">Smooth turn-taking</th>
  <th>User barge-in</th>
  <th colspan="3">Response quality</th>
</tr>
<tr>
  <th>TOR ↑</th>
  <th>Latency ↓</th>
  <th>TOR ↓</th>
  <th>Latency ↑</th>
  <th>Stop latency ↑</th>
  <th>Stop latency ↓</th>
  <th>Relevance ↑</th>
  <th>Consistency ↑</th>
  <th>TOR ↑</th>
  <th>Latency ↓</th>
  <th>Stop latency ↓</th>
  <th>Relevance ↑</th>
  <th>Consistency ↑</th>
  <th>QA Score ↑</th>
</tr>
</thead>
<tbody>
<tr><th colspan="15">English</th></tr>
<tr><td>DuplexCascade</td><td>69.6±5.6</td><td>4.92±0.43</td><td><strong>0.3</strong>±0.5</td><td><strong>7.58</strong>±0.28</td><td>0.41±0.25</td><td><strong>0.35</strong>±0.21</td><td>64.8±2.9</td><td>77.7±3.2</td><td>85.8±3.0</td><td>2.99±0.17</td><td>2.91±0.41</td><td>61.3±2.2</td><td>60.2±2.9</td><td>24.2±2.5</td></tr>
<tr><td>Freeze-Omni</td><td>80.0±4.9</td><td>2.66±0.30</td><td>61.8±4.8</td><td>1.90±0.48</td><td>2.21±0.45</td><td>1.71±0.39</td><td>59.4±2.9</td><td>68.7±3.3</td><td>85.4±3.0</td><td>2.24±0.19</td><td>2.43±0.26</td><td>56.0±2.3</td><td>53.5±2.6</td><td>17.0±2.7</td></tr>
<tr><td>BayLing-Duplex</td><td>88.8±3.9</td><td>1.67±0.14</td><td>33.3±4.6</td><td>2.05±0.27</td><td>1.68±0.59</td><td>1.69±0.58</td><td>70.0±3.4</td><td>60.4±3.7</td><td>91.7±2.4</td><td>1.89±0.12</td><td>1.76±0.28</td><td>61.9±2.2</td><td>51.4±2.9</td><td>17.5±2.6</td></tr>
<tr><td>Moshi</td><td>73.1±5.4</td><td>1.60±0.35</td><td>25.5±4.3</td><td>4.82±0.42</td><td>1.84±0.63</td><td>1.03±0.22</td><td>77.7±3.3</td><td>88.3±2.9</td><td>79.8±3.4</td><td>1.79±0.25</td><td>2.26±0.27</td><td>72.3±2.3</td><td>64.0±2.8</td><td>21.6±2.8</td></tr>
<tr><td>Moshi-RL</td><td><strong>95.8</strong>±2.5</td><td><strong>0.58</strong>±0.15</td><td>25.5±4.3</td><td>2.22±0.24</td><td>1.25±0.35</td><td>1.12±0.22</td><td><strong>82.7</strong>±3.1</td><td><strong>93.7</strong>±2.1</td><td><strong>94.3</strong>±2.0</td><td>1.02±0.18</td><td>1.95±0.24</td><td><strong>77.9</strong>±2.3</td><td>65.1±2.7</td><td>24.0±2.8</td></tr>
<tr><td>PersonaPlex</td><td>89.2±3.8</td><td>1.41±0.25</td><td>33.3±4.6</td><td>3.00±0.38</td><td>4.07±0.85</td><td>2.06±0.63</td><td>66.7±3.2</td><td>79.8±3.2</td><td>90.2±2.5</td><td>0.35±0.02</td><td><strong>1.28</strong>±0.16</td><td>76.5±2.3</td><td><strong>67.6</strong>±2.7</td><td>25.5±2.7</td></tr>
<tr><td>PersonaPlex-RL</td><td>93.1±3.1</td><td>0.97±0.24</td><td>36.5±4.7</td><td>1.61±0.20</td><td><strong>4.19</strong>±0.89</td><td>1.86±0.54</td><td>72.9±3.2</td><td>86.9±2.7</td><td>90.7±2.5</td><td><strong>0.32</strong>±0.05</td><td>1.50±0.20</td><td>77.7±2.3</td><td>66.8±2.7</td><td><strong>26.2</strong>±2.8</td></tr>
<tr><th colspan="15">Japanese</th></tr>
<tr><td>J-Moshi</td><td>31.5±4.9</td><td><strong>1.90</strong>±0.47</td><td><strong>22.8</strong>±4.5</td><td><strong>7.02</strong>±0.50</td><td>2.51±1.23</td><td><strong>2.35</strong>±0.52</td><td><strong>61.2</strong>±3.7</td><td><strong>91.7</strong>±2.0</td><td>77.9±3.6</td><td><strong>0.87</strong>±0.15</td><td><strong>2.04</strong>±0.27</td><td><strong>49.7</strong>±2.0</td><td><strong>63.9</strong>±2.5</td><td><strong>11.8</strong>±2.2</td></tr>
<tr><td>LLM-jp-Moshi</td><td><strong>84.0</strong>±3.9</td><td>2.26±0.29</td><td>69.1±5.0</td><td>0.87±0.36</td><td><strong>3.07</strong>±0.64</td><td>2.73±0.37</td><td>59.7±2.5</td><td>81.7±2.6</td><td><strong>84.1</strong>±3.2</td><td>1.87±0.22</td><td>2.33±0.28</td><td>45.4±2.0</td><td>57.5±2.3</td><td>10.5±2.0</td></tr>
</tbody>
</table>

</details>

<details>
<summary>Multi-turn evaluation (Reference-conditioned)</summary>

Multi-turn evaluation conditioned on the reference histories of both the user and the system.

<table>
<thead>
<tr>
  <th rowspan="3">Model</th>
  <th colspan="8">Chat (Real data)</th>
  <th colspan="6">Multi-turn QA (Synthetic data)</th>
</tr>
<tr>
  <th colspan="2">Smooth turn-taking</th>
  <th colspan="2">Pause handling</th>
  <th>User backchannel</th>
  <th>User barge-in</th>
  <th colspan="2">Response quality</th>
  <th colspan="2">Smooth turn-taking</th>
  <th>User barge-in</th>
  <th colspan="3">Response quality</th>
</tr>
<tr>
  <th>TOR ↑</th>
  <th>Latency ↓</th>
  <th>TOR ↓</th>
  <th>Latency ↑</th>
  <th>Stop latency ↑</th>
  <th>Stop latency ↓</th>
  <th>Relevance ↑</th>
  <th>Consistency ↑</th>
  <th>TOR ↑</th>
  <th>Latency ↓</th>
  <th>Stop latency ↓</th>
  <th>Relevance ↑</th>
  <th>Consistency ↑</th>
  <th>QA Score ↑</th>
</tr>
</thead>
<tbody>
<tr><th colspan="15">English</th></tr>
<tr><td>Moshi</td><td>61.5±6.0</td><td>1.11±0.31</td><td><strong>7.5</strong>±2.6</td><td><strong>7.34</strong>±0.36</td><td>3.04±0.57</td><td>1.75±0.38</td><td>66.9±3.2</td><td>90.6±2.4</td><td>91.9±2.3</td><td>0.41±0.09</td><td>2.23±0.26</td><td>72.8±2.4</td><td>70.5±2.6</td><td>26.6±3.0</td></tr>
<tr><td>Moshi-RL</td><td>92.3±3.3</td><td>0.72±0.17</td><td>17.8±3.8</td><td>2.76±0.26</td><td>2.41±0.50</td><td><strong>1.07</strong>±0.19</td><td>72.3±3.3</td><td><strong>94.8</strong>±2.0</td><td>94.1±2.0</td><td>0.34±0.05</td><td>1.82±0.24</td><td>71.9±2.5</td><td>69.7±2.5</td><td>26.8±2.9</td></tr>
<tr><td>PersonaPlex</td><td>93.1±3.1</td><td>0.65±0.12</td><td>26.3±4.3</td><td>3.76±0.36</td><td><strong>5.31</strong>±0.71</td><td>2.30±0.52</td><td>75.8±3.1</td><td>89.8±2.5</td><td>96.4±1.6</td><td>0.28±0.01</td><td><strong>1.05</strong>±0.13</td><td><strong>82.0</strong>±2.2</td><td>73.9±2.6</td><td><strong>33.0</strong>±3.1</td></tr>
<tr><td>PersonaPlex-RL</td><td><strong>97.7</strong>±1.8</td><td><strong>0.62</strong>±0.14</td><td>31.5±4.6</td><td>2.27±0.24</td><td>4.67±0.63</td><td>2.00±0.46</td><td><strong>77.5</strong>±3.2</td><td>89.2±2.5</td><td><strong>97.9</strong>±1.2</td><td><strong>0.27</strong>±0.03</td><td>1.19±0.16</td><td>81.6±2.2</td><td><strong>75.1</strong>±2.5</td><td>31.7±3.1</td></tr>
<tr><th colspan="15">Japanese</th></tr>
<tr><td>J-Moshi</td><td>61.6±5.1</td><td><strong>0.88</strong>±0.22</td><td>34.8±5.1</td><td><strong>5.06</strong>±0.50</td><td>1.98±0.63</td><td><strong>1.25</strong>±0.28</td><td><strong>63.0</strong>±2.8</td><td>87.1±2.5</td><td><strong>90.6</strong>±2.6</td><td><strong>0.31</strong>±0.04</td><td><strong>1.42</strong>±0.19</td><td>50.9±1.8</td><td>60.8±2.3</td><td><strong>14.6</strong>±2.3</td></tr>
<tr><td>LLM-jp-Moshi</td><td><strong>88.3</strong>±3.4</td><td>1.34±0.21</td><td><strong>26.7</strong>±4.8</td><td>4.50±0.39</td><td><strong>2.25</strong>±0.53</td><td>1.86±0.32</td><td>62.2±2.7</td><td><strong>90.0</strong>±2.2</td><td>90.4±2.6</td><td>0.69±0.12</td><td>1.59±0.21</td><td><strong>51.4</strong>±1.7</td><td><strong>61.3</strong>±2.4</td><td>14.3±2.3</td></tr>
</tbody>
</table>

</details>

## Documents 
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

