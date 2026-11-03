# Anonymous Artifact

A CLI implementation of three-stage semantic filtering. Given a query and a document, it returns a binary prediction and the stage that produced it.

| Stage | Input to the model | How it runs |
| --- | --- | --- |
| Stage 1 | Selected evidence from the document | Several records packed into one SLM prompt |
| Stage 2 | The full document | One SLM call per record in the maximal Stage-2 pool |
| Stage 3 | The full document | One Oracle call per unresolved record |

SLM means the small language model. Oracle means the stronger model selected for calibration and fallback. Stage 2 and the Oracle use the same full-document inference function.

Neighbor Support (NS) adjusts each SLM stage's score using nearby predictions. WS&R jointly selects the two acceptance thresholds from one calibration sample: fixed-sequence scanning within each Stage-1 threshold track, with Bonferroni across tracks.

The workflow produces `runs/final/predictions.jsonl` and `runs/final/summary.json`. The supplied data is a three-record example; use your own query–document population for an actual experiment.

## Before running

Use Linux or WSL2, Python 3.10+, a CUDA-enabled PyTorch installation, and a GPU with enough memory for the selected models. The commands use Bash syntax.

All parameters are CLI arguments. Choose the models, score parameters, grids, calibration budget, accuracy target, and failure budget before requesting calibration Oracle labels. The examples use:

| Setting | Value used below |
| --- | --- |
| Embedding model | `Qwen/Qwen3-Embedding-4B` |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` |
| SLM and its tokenizer | `Qwen/Qwen3-4B-Instruct-2507` |
| SLM endpoint | `http://localhost:8000/v1` |
| Oracle | Replace `YOUR_ORACLE_MODEL` with your served model name |
| Generation temperature | 0 |
| Margin temperature | 15 |
| NS neighbors / weight | 20 / 1 |
| Stage-1 / Stage-2 grid points | Up to 32 / 64 |
| Target agreement / failure probability | `tau=0.9` / `delta=0.05` |

The Oracle commands use a public chat API base URL as an example. Replace it for your provider. Both chat endpoints must support `max_completion_tokens`; the SLM endpoint must additionally return paired label probabilities through vLLM's `logprob_token_ids` option.

## 1. Install and prepare the data — Terminal A

Extract the archive, then run:

```bash
unzip anonymous-artifact.zip
cd anonymous-artifact
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
mkdir -p data
cp examples/records.jsonl data/records.jsonl
```

If the files are already extracted, begin inside the directory containing `rrf.py` and `requirements.txt`. Run every command below from that directory.

Keep Terminal A in `.venv`. It runs the six scripts. Terminal B, opened later, runs only the SLM server in a separate environment.

### Input format

`data/records.jsonl` is a UTF-8 file with one JSON object per line:

```json
{"id":"review-1","query":"Does this review complain about the wait time?","document":"We booked a table for seven. We waited an hour before being seated. The food was good.","label":"yes"}
{"id":"review-2","query":"Does this review complain about the wait time?","document":"The staff seated us immediately. Dinner was excellent. We will come back.","label":"no"}
```

| Field | Requirement |
| --- | --- |
| `id` | A unique, nonempty string for this query–document pair; sequential IDs are sufficient |
| `query` | The complete predicate, with any claim, name, or specification filled in |
| `document` | The complete, nonempty source text |
| `label` | Optional benchmark ground truth, exactly `yes` or `no`; supply it for all records or omit it for all records |

Give different IDs to the same document evaluated against different queries. Keep the full document in `document`; evidence is generated automatically.

The optional benchmark `label` is used only for final accuracy and Macro-F1. Calibration uses labels requested from the Oracle later in the workflow.

## 2. Retrieve evidence candidates — Terminal A

Reads `data/records.jsonl`. Writes `runs/rrf.jsonl`.

```bash
python rrf.py \
  --input data/records.jsonl \
  --output runs/rrf.jsonl \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 16 \
  --window-size 3 \
  --rrf-k 60 \
  --buffer 5
```

For each document, BlingFire creates sentence windows of size 3 with stride 1. BM25 and embedding similarity rank those windows within that document. RRF combines the ranks and keeps the prefix ending at the largest score gap, plus five windows.

The output retains every input record and adds sentences, candidate windows, and retrieval scores.

## 3. Rerank and merge evidence — Terminal A

Reads `runs/rrf.jsonl`. Writes `runs/evidence.jsonl`.

```bash
python rerank.py \
  --input runs/rrf.jsonl \
  --output runs/evidence.jsonl \
  --reranker-model Qwen/Qwen3-Reranker-0.6B \
  --device cuda \
  --batch-size 8 \
  --max-chunks 5
```

The reranker scores the surviving windows. Selection ends at the largest reranker-score gap and is capped at five windows. Their sentences are merged in document order, removing overlapping sentence indices. If all scores tie, selection keeps the available prefix up to the cap.

The output adds `evidence`, `selected_sentence_indices`, and `selected_chunks`. It preserves the full `document` for later inference.

## 4. Draw the calibration sample — Terminal A

Reads `runs/evidence.jsonl`. Writes the files under `data/split/`.

```bash
python split.py \
  --input runs/evidence.jsonl \
  --output-dir data/split \
  --seed 42
```

The default sample size is `min(N, max(500, ceil(0.01*N)))`. To choose a fixed budget, add `--size 1000`, with a value between 1 and N. The sample is drawn uniformly without replacement, and its file preserves the random draw order.

| File | Meaning | Used by |
| --- | --- | --- |
| `data/split/population.jsonl` | All N records in original order | Stage 1, threshold search, routing, and final assembly |
| `data/split/calibration.jsonl` | Calibration subset S, still part of the population | Calibration Oracle calls and threshold search |
| `data/split/remaining.jsonl` | Population records outside S | Optional inspection of the sample complement |
| `data/split/split.json` | Sample sizes and random seed | Inspection |

The certified population includes the calibration records. Later commands use `population.jsonl` as their population input. With the three-record example, the default calibration sample contains all three records.

## 5. Prepare the two NS vector files — Terminal A

Reads `data/split/population.jsonl`. Writes both NPZ files below.

```bash
python ns.py embed \
  --input data/split/population.jsonl \
  --evidence-output runs/evidence_vectors.npz \
  --document-output runs/document_vectors.npz \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 16
```

`evidence_vectors.npz` contains embeddings of query + evidence. `document_vectors.npz` contains embeddings of query + full document. Each file stores record IDs and normalized vectors.

This command prepares vectors. NS scores are computed after the corresponding SLM predictions exist. Stage-2 NS later searches only within its maximal Stage-2 pool.

Complete the local retrieval, reranking, and vector commands before starting the SLM server on the same GPU. These commands release their model memory when they exit; NS scoring uses the saved vectors on CPU.

## 6. Start the SLM server — Terminal B

Open Terminal B at the artifact root. Keep Terminal A available for the next commands.

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -U vllm
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 \
  --max-model-len 32768 \
  --generation-config vllm
```

Wait for the server to report that it is ready to accept requests. Leave this terminal running during both SLM stages.

The client `--model` value must match `--served-model-name`. The client `--tokenizer` must match the model behind that endpoint. If your server uses authentication, set its key in Terminal A and pass `--api-key-env SLM_API_KEY` to the SLM inference commands.

## 7. Run Stage 1, then compute NS₁ — Terminal A

First run batched inference over the complete population:

```bash
python inference.py stage1 \
  --input data/split/population.jsonl \
  --output runs/stage1.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://localhost:8000/v1 \
  --temperature 0 \
  --seed 42 \
  --batch-token-budget 8192 \
  --context-window 32768 \
  --score-temperature 15
```

Stage 1 puts several query–evidence records into each prompt with DocID delimiters. The 8,192-token input budget includes instructions and the chat template. Completion space is reserved separately within the 32,768-token context limit.

At each output label, the code reads Y and N probabilities using matching whitespace tokenization. Malformed output, missing paired probabilities, or evidence exceeding the input budget causes deferral.

Next compute neighbor support and freeze the Stage-1 grid:

```bash
python ns.py score \
  --stage stage1 \
  --input data/split/population.jsonl \
  --predictions runs/stage1.jsonl \
  --vectors runs/evidence_vectors.npz \
  --output runs/stage1_scores.jsonl \
  --grid runs/grid1.json \
  --stage2-input runs/stage2_input.jsonl \
  --k 20 \
  --gamma 1 \
  --grid-size 32
```

| Output | Contents |
| --- | --- |
| `runs/stage1.jsonl` | Predictions, base scores, and token usage |
| `runs/stage1_scores.jsonl` | Predictions with NS and final routing scores |
| `runs/grid1.json` | Up to 32 unique routing-score quantiles, forming Λ₁ |
| `runs/stage2_input.jsonl` | Maximal Stage-2 pool |

The maximal pool contains records with a Stage-1 score below `max(Λ₁)`, plus invalid Stage-1 records. It covers every record that any candidate Stage-1 threshold could defer.

## 8. Run Stage 2, then compute NS₂ — Terminal A

Read the full document for each record in the maximal Stage-2 pool:

```bash
python inference.py full \
  --input runs/stage2_input.jsonl \
  --output runs/stage2.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://localhost:8000/v1 \
  --temperature 0 \
  --seed 42 \
  --context-window 32768 \
  --score-temperature 15
```

Here `full` selects full-document input. The `--input` file determines which records receive calls. Each record receives one call.

Compute NS₂ within this fixed maximal pool and freeze Λ₂:

```bash
python ns.py score \
  --stage stage2 \
  --input runs/stage2_input.jsonl \
  --predictions runs/stage2.jsonl \
  --vectors runs/document_vectors.npz \
  --output runs/stage2_scores.jsonl \
  --grid runs/grid2.json \
  --k 20 \
  --gamma 1 \
  --grid-size 64
```

The output grid contains up to 64 unique quantiles. Both stages' predictions, NS scores, and grids are now fixed. Threshold search reuses these stored results for every candidate pair.

The SLM work is complete. Stop the server with Ctrl+C in Terminal B. Continue in Terminal A.

## 9. Obtain Oracle labels for the calibration sample — Terminal A

Set the credential:

```bash
export ORACLE_API_KEY="your-key"
```

Replace `YOUR_ORACLE_MODEL` with the actual served model name. Use the same Oracle model, endpoint, and decoding options for calibration and residual calls.

```bash
python inference.py full \
  --oracle \
  --input data/split/calibration.jsonl \
  --output runs/oracle_calibration.jsonl \
  --model YOUR_ORACLE_MODEL \
  --base-url https://api.openai.com/v1 \
  --api-key-env ORACLE_API_KEY \
  --temperature 0 \
  --max-output-tokens 16
```

The `--oracle` flag selects binary-label parsing and skips SLM score extraction. The prompt contains the query and full document; the response must be exactly `Y` or `N`. The output contains predictions and measured token usage.

Use a non-thinking mode for this short completion budget. Append `--reasoning-effort none` if your provider supports that option. For a vLLM model with a thinking chat template, use `--disable-thinking`. Apply the same option to the residual Oracle command.

## 10. Fit the two thresholds — Terminal A

Reads the fixed predictions, scores, grids, population, calibration sample, and calibration Oracle labels:

```bash
python wsr.py fit \
  --population data/split/population.jsonl \
  --stage1 runs/stage1_scores.jsonl \
  --stage2 runs/stage2_scores.jsonl \
  --grid1 runs/grid1.json \
  --grid2 runs/grid2.json \
  --calibration data/split/calibration.jsonl \
  --oracle-labels runs/oracle_calibration.jsonl \
  --output-dir runs/search \
  --tau 0.9 \
  --delta 0.05 \
  --slm-input-price 0.2 \
  --slm-output-price 0.2 \
  --oracle-input-price 5 \
  --oracle-output-price 30 \
  --oracle-output-tokens 2
```

Prices are examples per million tokens. Choose the prices and expected Oracle output length before calibration, using the models you actually select.

For each fixed λ₁, the search scans λ₂ from largest to smallest. It certifies a contiguous prefix and stops that track at its first failure. Every candidate in a track uses the level `delta / number_of_lambda1_tracks`. The selected pair has the lowest estimated cost among the certified pairs.

| Output | Contents |
| --- | --- |
| `runs/search/policy.json` | Selected thresholds, frozen grids, calibration IDs, assumptions, and fallback status |
| `runs/search/candidates.jsonl` | Visited candidates, certification decisions, routing counts, and estimated costs |

If no candidate is certified, `policy.json` has `selected: null` and `fallback: "all_oracle"`. Routing then resolves the population with Oracle labels. Small calibration samples can produce this result.

## 11. Route the population — Terminal A

```bash
python wsr.py route \
  --population data/split/population.jsonl \
  --stage1 runs/stage1_scores.jsonl \
  --stage2 runs/stage2_scores.jsonl \
  --policy runs/search/policy.json \
  --oracle-labels runs/oracle_calibration.jsonl \
  --output-dir runs/route
```

Each record takes the first qualifying path:

```text
Stage-1 score >= lambda1 → use the Stage-1 prediction
otherwise, Stage-2 score >= lambda2 → use the Stage-2 prediction
otherwise → use the Oracle label
```

Routing reuses the fitted scores and the same acceptance calculation as fitting. It checks the population, maximal Stage-2 pool, grids, and calibration IDs. A deferred record with an existing calibration Oracle label is resolved from that label immediately.

| Output | Contents |
| --- | --- |
| `runs/route/accepted.jsonl` | Results already resolved by an SLM stage or a stored calibration Oracle label |
| `runs/route/pending_oracle.jsonl` | Full input records that still require an Oracle call |

`data/split/remaining.jsonl` is the calibration-sample complement. `runs/route/pending_oracle.jsonl` is the final unresolved set after routing and calibration-label reuse. The residual Oracle command reads `pending_oracle.jsonl`.

## 12. Complete the Oracle residual and assemble results — Terminal A

Use the same Oracle identity and decoding settings as the calibration call:

```bash
python inference.py full \
  --oracle \
  --input runs/route/pending_oracle.jsonl \
  --output runs/oracle_residual.jsonl \
  --model YOUR_ORACLE_MODEL \
  --base-url https://api.openai.com/v1 \
  --api-key-env ORACLE_API_KEY \
  --temperature 0 \
  --max-output-tokens 16

python wsr.py merge \
  --population data/split/population.jsonl \
  --accepted runs/route/accepted.jsonl \
  --oracle runs/oracle_residual.jsonl \
  --output-dir runs/final
```

An empty pending file produces an empty Oracle output without contacting a model. Final assembly checks that resolved results and new Oracle results partition the population.

| Final file | Contents |
| --- | --- |
| `runs/final/predictions.jsonl` | One prediction per population record, in original order |
| `runs/final/summary.json` | Stage counts; accuracy and Macro-F1 when all records have benchmark labels |

Each prediction row has this format:

```json
{"id":"review-1","prediction":"yes","stage":"stage1"}
```

The `stage` value is `stage1`, `stage2`, or `oracle`. Stage counts sum to N. Accuracy is the fraction of predictions matching the optional benchmark labels. Macro-F1 averages class-wise F1 over labels present in the references or predictions, including single-class populations.

## Parameter meanings

Each script provides its arguments through `--help`.

| Argument | Meaning |
| --- | --- |
| `--model` | Model name recognized by the chat server |
| `--tokenizer` | Matching tokenizer ID or local directory; defaults to `--model` |
| `--embedding-model`, `--reranker-model` | Model ID or local model directory for the corresponding local command |
| `--base-url` | Chat API base URL including `/v1` |
| `--api-key-env` | Name of the environment variable holding the credential |
| `--batch-size` | Local embedding/reranking batch size in `rrf.py`, `rerank.py`, or `ns.py embed` |
| `--batch-token-budget` | Stage-1 packed-prompt input budget; default 8,192 |
| `--context-window` | SLM input plus completion limit; default 32,768 |
| `--temperature` | Generation temperature, enforced at 0 |
| `--score-temperature` | Margin transformation temperature T; default 15 |
| `--max-output-tokens` | Full-document completion budget; default 16 |
| `--size` | Fixed calibration size; default `min(N, max(500, ceil(0.01*N)))` |
| `--k`, `--gamma`, `--epsilon` | NS neighbors, adjustment weight, log-odds smoothing; defaults 20, 1, 0.001 |
| `--grid-size` | Number of quantile points before duplicate removal; defaults 32 for Stage 1 and 64 for Stage 2 |
| `--tau`, `--delta` | Target agreement with the Oracle and total failure probability |

Changing inputs, models, scoring settings, or grids requires rebuilding the affected outputs and downstream files. Freeze the candidate family before requesting the calibration Oracle labels. Use the unchanged score and label files when applying a fitted policy.

## Scoring and certification definitions

The base score uses a signed margin:

```text
margin     = log p(Y) - log p(N)
prediction = yes if margin > 0, otherwise no
b          = sigmoid(margin / T)
```

A confident negative prediction has `b < 0.5`. The score represents the signed Y/N preference. The implementation compares paired label tokens at the same output position and with the same whitespace form.

NS uses top-K cosine neighbors within the stage's own pool, excludes self, and gives invalid neighbor predictions zero voting weight:

```text
w_ij        = max(cosine(v_i, v_j), 0)
W_i         = sum_j w_ij
NS_i        = weighted fraction of neighbors agreeing with prediction_i
rho_i       = W_i / (W_i + median(W))
shrunk_NS_i = rho_i * NS_i + (1 - rho_i) * 0.5
score_i     = log((b_i + epsilon_log) / (1 - b_i + epsilon_log))
              + gamma * (2 * shrunk_NS_i - 1)
```

Here `epsilon_log` is the `ns.py score --epsilon` smoothing parameter, default 0.001. Zero neighbor mass gives neutral support; a zero shrinkage denominator sets rho to 0. Stage 1 uses the whole population as its pool; Stage 2 uses the fixed maximal Stage-2 pool. FAISS performs exact cosine search in blocks, so compute grows quadratically with pool size.

For calibration, a record has loss 1 when an accepted SLM prediction disagrees with the Oracle. Oracle fallback has loss 0. The residual-centred betting calculation is:

```text
error_target = 1 - tau
q_t     = clip((N * error_target - sum_previous_errors) / (N - t + 1), 0, 1)
bet_t   = clip((q_t - mu_hat) / (variance_hat + (q_t - mu_hat)^2),
               0, 0.5 / (1 - q_t))
M_n     = product_t (1 + bet_t * (q_t - error_t))
```

The mean and variance estimates use previous errors only. At q_t = 1, the bet's upper cap is unbounded. A candidate passes when `M_n >= number_of_lambda1_tracks / delta`. Certification targets agreement with the fixed Oracle on this finite population; optional benchmark labels provide a separate evaluation.

Candidate selection uses measured SLM token usage and estimated Oracle costs. Oracle input lengths use the SLM tokenizer; output lengths use `--oracle-output-tokens`. Search-pass and calibration-labeling costs are reported separately. Retrieval/indexing compute and the final provider bill are outside this estimate.

Public interface and formula references: [embedding model](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), [vLLM log probabilities](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/), [aGRAPA definition](https://par.nsf.gov/servlets/purl/10379214), [Macro-F1 definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

## Common execution issues

| Symptom | Action |
| --- | --- |
| Chat endpoint connection fails | Keep Terminal B running and match the client base URL and served model name |
| Local embedding/reranking runs out of GPU memory | Reduce that command's `--batch-size`; complete local model work before starting vLLM |
| Full document exceeds the SLM context | Choose a supported longer context in both the server and client before running the population |
| Many SLM rows have missing paired probabilities | Use a vLLM endpoint supporting `logprob_token_ids` and a tokenizer matching its served model |
| Stage-2 IDs differ from the maximal pool | Regenerate the maximal pool and its downstream predictions, scores, grids, and policy together |
| No threshold pair is certified | Inspect `policy.json`; full Oracle fallback is the defined outcome |
| Final metrics reject benchmark labels | Supply valid labels for every record, or omit benchmark labels entirely |

## Files

```text
anonymous-artifact/
├── .gitignore
├── README.md
├── requirements.txt
├── examples/
│   └── records.jsonl
├── rrf.py
├── rerank.py
├── split.py
├── inference.py
├── ns.py
└── wsr.py
```

The six scripts are independent entry points and import no other project script. They exchange data through explicit CLI input/output paths. Each script contains its own small I/O and model helpers. Generated data, vectors, and results live under `data/` and `runs/`.

The anonymous distribution contains these source files and the synthetic example. It excludes Git metadata, credentials, local environments, model weights, and generated runs.
