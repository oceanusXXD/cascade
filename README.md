# CascadeFilter

Research implementation of **Pay Only What You Need: Certifiably Accurate, Cost-Optimal LLM Semantic Filtering via Adaptive Cascade Routing**.

CascadeFilter evaluates a natural-language predicate on each document using three stages:

| Stage | Model input | Invocation |
| --- | --- | --- |
| 1 | Retrieved, reranked evidence | Multiple records in one SLM prompt |
| 2 | Full document | One SLM call per record |
| 3 | Full document | One Oracle call per residual record |

Each SLM stage produces a prediction and a confidence. Neighbor Support (NS) adjusts that confidence using nearby predictions. A single calibration sample then certifies the two acceptance thresholds jointly using WS&R, with **fixed-sequence scanning within each λ₁ track and Bonferroni across tracks**.

All settings are command-line arguments. The commands below are the complete workflow, in execution order.

```text
records → RRF → rerank → calibration sampling → two embedding views
        → Stage 1 → NS₁ / Λ₁ / maximal Stage-2 pool
        → Stage 2 → NS₂ / Λ₂
        → Oracle labels for calibration → WS&R threshold search
        → routing → Oracle on the residual → final predictions
```

## 1. Install and prepare the input

Use Linux or WSL2 for the vLLM server, Python 3.10+, and a CUDA-capable GPU with sufficient memory. The paper uses an A100 80 GB. Embedding and reranking run locally; the SLM runs behind a vLLM endpoint; the Oracle uses a chat-completions-compatible endpoint.

Clone the reproduction branch, which contains this workflow:

```bash
git clone --branch refactor/simple-reproduction https://github.com/oceanusXXD/cascade.git
cd cascade
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

The local embedding and reranker commands use the CUDA-enabled PyTorch installation in this environment. Run all subsequent commands from the repository root. Each command creates its output directories as needed.

The included data contains three example records. To follow the workflow with those records:

```bash
mkdir -p data
cp examples/records.jsonl data/records.jsonl
```

For your experiment, put the actual dataset at `data/records.jsonl`. Each UTF-8 JSONL line represents one query–document pair:

```json
{"id":"q1-d1","query":"Does this review complain about the wait time?","document":"We waited an hour before being seated. The food was good.","label":"yes"}
{"id":"q1-d2","query":"Does this review complain about the wait time?","document":"The staff seated us immediately. Dinner was excellent.","label":"no"}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Unique, nonempty string for the query–document pair |
| `query` | Yes | Complete predicate, with any claim, function name, or specification filled in |
| `document` | Yes | Complete, nonempty source text |
| `label` | Optional | Benchmark ground truth, `yes` or `no`, used for final evaluation |

Sequential IDs are sufficient. Give different IDs to the same document evaluated against different queries. The six paper datasets are supplied separately; the included records illustrate the schema and workflow. A tiny population may select full Oracle fallback.

## 2. Retrieve candidates: BM25 + embedding RRF

```bash
python -m cascade.rrf \
  --input data/records.jsonl \
  --output runs/rrf.jsonl \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 16 \
  --window-size 3 \
  --rrf-k 60 \
  --buffer 5
```

Within each document, BlingFire splits sentences into overlapping three-sentence windows with stride 1. BM25 and cosine embedding scores rank these windows; reciprocal rank fusion combines their ranks. The prefix ending at the largest RRF-score gap, plus five windows, becomes the reranker input.

Output: `runs/rrf.jsonl`, with the original record, sentences, and candidate windows with their three scores.

## 3. Rerank and select evidence

```bash
python -m cascade.rerank \
  --input runs/rrf.jsonl \
  --output runs/evidence.jsonl \
  --reranker-model Qwen/Qwen3-Reranker-0.6B \
  --device cuda \
  --batch-size 8 \
  --max-chunks 5
```

The cross-encoder reranks the candidates. Select the prefix ending at the largest reranker-score gap, cap it at five windows, and merge sentences in document order, removing overlapping sentence indices. When all scores tie, retain the available prefix up to the cap.

Output: `runs/evidence.jsonl`, including `evidence`, `selected_sentence_indices`, and `selected_chunks`. The full `document` remains available.

## 4. Draw the calibration sample

```bash
python -m cascade.split \
  --input runs/evidence.jsonl \
  --output-dir data/split \
  --seed 42
```

The default calibration size is `min(N, max(500, ceil(0.01*N)))`. To use a fixed sample budget, add `--size 1000`, with a value no larger than N. Fix that budget before requesting Oracle labels.

| Output | Contents |
| --- | --- |
| `data/split/population.jsonl` | All N records, in original order |
| `data/split/calibration.jsonl` | Uniform sample without replacement, in random draw order |
| `data/split/remaining.jsonl` | Complement of the calibration sample |
| `data/split/split.json` | Population size, calibration size, and seed |

The calibration sample belongs to the certified population. All Stage-1 inference and WS&R population counts use `population.jsonl`. The next steps use predictions and text; Oracle labels are requested in Step 9.

## 5. Prepare both NS embedding views

```bash
python -m cascade.ns embed \
  --input data/split/population.jsonl \
  --evidence-output runs/evidence_vectors.npz \
  --document-output runs/document_vectors.npz \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 16
```

Each NPZ contains IDs and normalized vectors. Stage 1 uses query + evidence vectors; Stage 2 uses query + full-document vectors. Its neighbor search later selects only records in the fixed maximal Stage-2 pool.

Complete Steps 2–5 before starting the SLM server on the same GPU. Each local model command releases its GPU memory when it exits. Subsequent NS scoring reads these vectors and runs on CPU.

## 6. Start the SLM server in a second terminal

Keep the first terminal in `.venv`. Open the second terminal at the cloned repository root and create a separate vLLM environment:

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

Keep the server running through Step 8.

The SLM endpoint must expose token log probabilities and vLLM's `logprob_token_ids` option. Both chat endpoints must accept `max_completion_tokens`. Model interfaces follow the official [Qwen embedding](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [Qwen reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), and [vLLM log-probability](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/) documentation.

## 7. Run Stage 1 and compute NS₁

Back in the first terminal:

```bash
python -m cascade.inference stage1 \
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

python -m cascade.ns score \
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

Stage 1 packs several query–evidence pairs into one prompt with DocID delimiters. The 8,192-token input budget includes the chat template and instructions. Output space is reserved separately within the SLM context limit. Malformed output, missing paired label probabilities, or an oversized evidence record causes deferral.

NS uses predictions from the same stage, excludes self, and weights agreement by nonnegative cosine similarity. It shrinks support toward 0.5 using median neighborhood mass, then adjusts the confidence in log-odds space. NS reads no benchmark or Oracle labels.

Outputs:

- `runs/stage1.jsonl`: predictions, base confidence, and token usage.
- `runs/stage1_scores.jsonl`: predictions with NS and routing scores.
- `runs/grid1.json`: up to 32 unique score quantiles.
- `runs/stage2_input.jsonl`: records below the largest Stage-1 threshold, plus invalid Stage-1 records; this is the maximal Stage-2 pool.

## 8. Run Stage 2 and compute NS₂

```bash
python -m cascade.inference full \
  --input runs/stage2_input.jsonl \
  --output runs/stage2.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://localhost:8000/v1 \
  --temperature 0 \
  --seed 42 \
  --context-window 32768 \
  --score-temperature 15

python -m cascade.ns score \
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

Stage 2 performs one full-document call per record in the maximal pool. Its NS neighbors and median mass are computed within that fixed pool. `grid2.json` contains up to 64 unique score quantiles.

This pass supplies every candidate's Stage-2 result. Threshold search reuses these results, so it needs no further SLM calls. Both score files and grids are now fixed. You can stop the SLM server with Ctrl+C in its terminal.

## 9. Label the calibration sample with the Oracle

Set the credential in the first terminal:

```bash
export ORACLE_API_KEY="your-key"
```

Replace `YOUR_ORACLE_MODEL` with your provider's actual model name in this command and the residual Oracle command in Step 12. For a different provider, replace the base URL in both commands too.

```bash
python -m cascade.inference full \
  --oracle \
  --input data/split/calibration.jsonl \
  --output runs/oracle_calibration.jsonl \
  --model YOUR_ORACLE_MODEL \
  --base-url https://api.openai.com/v1 \
  --api-key-env ORACLE_API_KEY \
  --temperature 0 \
  --max-output-tokens 16
```

Stage 2 and the Oracle share `full_inference` and the same query/full-document prompt. The `--oracle` flag selects binary-label parsing and skips SLM confidence extraction. The response must be exactly `Y` or `N`.

Use a non-thinking Oracle mode for this short completion budget. If the provider supports it, append `--reasoning-effort none` to **both** Oracle commands. For a vLLM model with a thinking chat template, `--disable-thinking` sends `enable_thinking=false`. Use the same model and decoding options for calibration and residual calls.

Output: `runs/oracle_calibration.jsonl`, with Oracle predictions and token usage. The input `label` field remains reserved for final evaluation.

## 10. Select thresholds with WS&R

```bash
python -m cascade.wsr fit \
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

Prices are examples in currency units per million tokens. Replace them with the prices for the models used. Fix `--tau`, `--delta`, prices, and expected Oracle output length before requesting calibration labels; those arguments appear together here because this command consumes them.

For each λ₁, scan λ₂ from largest to smallest. An accepted SLM prediction contributes loss 1 when it disagrees with the Oracle; fallback contributes loss 0. Evaluate WS&R betting wealth at `epsilon = 1 - tau`, with residual-population centering and clipped, predictable aGRAPA-style bets. Each track uses the level `delta / number_of_lambda1_tracks`; stop at the first failed candidate. Select the cheapest certified pair. An empty certified set selects full Oracle fallback.

Outputs:

- `runs/search/policy.json`: selected thresholds, target, track level, price assumptions, and fallback status.
- `runs/search/candidates.jsonl`: candidates visited during the fixed-sequence scans, with certification results and estimated costs.

## 11. Apply the selected thresholds

```bash
python -m cascade.wsr route \
  --population data/split/population.jsonl \
  --stage1 runs/stage1_scores.jsonl \
  --stage2 runs/stage2_scores.jsonl \
  --policy runs/search/policy.json \
  --oracle-labels runs/oracle_calibration.jsonl \
  --output-dir runs/route
```

Each record takes the first qualifying path:

```text
Stage-1 score >= lambda1 → Stage-1 prediction
otherwise, Stage-2 score >= lambda2 → Stage-2 prediction
otherwise → Oracle
```

Outputs: `runs/route/accepted.jsonl` and `runs/route/pending_oracle.jsonl`. A deferred record with an existing calibration Oracle label is resolved immediately using that stored label.

## 12. Run the residual Oracle calls and merge

Use the same Oracle model, endpoint, and decoding options as Step 9:

```bash
python -m cascade.inference full \
  --oracle \
  --input runs/route/pending_oracle.jsonl \
  --output runs/oracle_residual.jsonl \
  --model YOUR_ORACLE_MODEL \
  --base-url https://api.openai.com/v1 \
  --api-key-env ORACLE_API_KEY \
  --temperature 0 \
  --max-output-tokens 16

python -m cascade.wsr merge \
  --population data/split/population.jsonl \
  --accepted runs/route/accepted.jsonl \
  --oracle runs/oracle_residual.jsonl \
  --output-dir runs/final
```

An empty pending file produces an empty Oracle output without contacting a model.

The final deliverables are:

| File | Contents |
| --- | --- |
| `runs/final/predictions.jsonl` | One prediction per record, in original population order |
| `runs/final/summary.json` | Stage counts; accuracy and binary macro-F1 when all input records have `label` |

Example final row:

```json
{"id":"q1-d1","prediction":"yes","stage":"stage1"}
```

## CLI reference

Each entry point exposes its arguments through `--help`. The important controls are:

| Command | Arguments and defaults |
| --- | --- |
| `rrf` | `--embedding-model Qwen/Qwen3-Embedding-4B`, `--device cuda`, `--batch-size 16`, `--max-length 32768`, `--window-size 3`, `--rrf-k 60`, `--buffer 5` |
| `rerank` | `--reranker-model Qwen/Qwen3-Reranker-0.6B`, `--device cuda`, `--batch-size 8`, `--max-length 8192`, `--max-chunks 5` |
| `split` | `--seed 42`; optional `--size` overrides the population-dependent sample size |
| `ns embed` | `--embedding-model`, `--device cuda`, `--batch-size 16`, `--max-length 32768` |
| `ns score` | `--k 20`, `--gamma 1`, `--epsilon 0.001`; `--grid-size` defaults to 32/64 for Stage 1/2 |
| `inference` | Required `--model` and `--base-url`; tokenizer defaults to model; `--temperature 0`, `--score-temperature 15`, `--batch-token-budget 8192`, `--context-window 32768`, `--max-output-tokens 16` |
| `wsr fit` | `--tau 0.9`, `--delta 0.05`, four token-price arguments, `--oracle-output-tokens 2` |

Model and tokenizer arguments accept a Hugging Face ID or the appropriate local model path. For a local model served under a short name, use that served name in `--model` and its matching model path in `--tokenizer`. Use `--api-key-env SLM_API_KEY` for an authenticated SLM endpoint after setting that variable. Oracle mode defaults to `ORACLE_API_KEY` when `--api-key-env` is omitted.

## Method and accounting details

- **Confidence:** this implementation uses `sigmoid(abs(log p(Y) - log p(N)) / T)`, confidence in the predicted label. The manuscript's §3.2.3 prints a signed margin; align that definition when matching the method formula by formula.
- **NS:** similarity-weighted agreement shrinks toward 0.5 with `rho = W / (W + median(W))`; routing adds `gamma * (2 * shrunk_NS - 1)` to smoothed confidence log odds. Invalid neighbor predictions carry zero voting weight. FAISS performs exact cosine search in blocks; compute grows quadratically with pool size.
- **Betting:** the residual target is `q_t = clip((N*epsilon - sum_previous_errors)/(N-t+1), 0, 1)`. The positive bet uses `(q_t - mu_hat)/(variance_hat + (q_t - mu_hat)^2)`, clipped to `[0, 0.5/(1-q_t)]`. Only previous errors enter its estimates. See [Waudby-Smith and Ramdas, Appendix B.3](https://par.nsf.gov/servlets/purl/10379214) for the underlying aGRAPA rule.
- **Cost:** candidate selection uses measured SLM usage and estimates Oracle input lengths with the SLM tokenizer and output lengths with `--oracle-output-tokens`. Maximal Stage-2 search and calibration-labeling costs are reported separately. Retrieval/indexing compute and the final provider bill are outside this estimate.
- **Target:** certification concerns agreement with the fixed Oracle on this finite population. Benchmark accuracy and macro-F1 use the optional `label`. Keep models, decoding, score parameters, grids, sample budget, tau, and delta fixed before observing calibration labels.

## File tree

```text
cascade/
├── README.md
├── requirements.txt
├── examples/
│   └── records.jsonl
└── cascade/
    ├── __init__.py
    ├── data.py          # Shared JSON/JSONL I/O
    ├── models.py        # Small model clients with explicit arguments
    ├── rrf.py           # BM25 + embedding fusion
    ├── rerank.py        # Reranking and evidence merging
    ├── split.py         # Calibration sampling
    ├── inference.py     # Batch evidence / shared full-document inference
    ├── ns.py            # Embeddings, NS, and quantile grids
    └── wsr.py           # Certification, routing, and final assembly
```

Generated inputs, vectors, and results live under `data/` and `runs/`.
