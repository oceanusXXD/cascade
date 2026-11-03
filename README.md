# CascadeFilter

Core implementation of **Pay Only What You Need: Certifiably Accurate, Cost-Optimal LLM Semantic Filtering via Adaptive Cascade Routing**.

CascadeFilter answers a natural-language predicate for each document through three stages:

1. **Stage 1 — batched SLM on evidence.** BM25 and embedding retrieval find short evidence windows; a reranker selects and merges them. Several query–evidence pairs share one SLM prompt.
2. **Stage 2 — SLM on the full document.** Each deferred document receives its own call.
3. **Stage 3 — Oracle on the full document.** Remaining records use the same full-document inference function with a different model.

Neighbor Support (NS) adjusts each SLM confidence using nearby predictions. WS&R jointly selects the two acceptance thresholds using **fixed-sequence certification within each λ₁ track and Bonferroni across tracks**.

## Repository

```text
cascade/
├── README.md
├── requirements.txt
├── config.json                 # Models, endpoints, method parameters, prices
├── examples/
│   └── records.jsonl           # Three records illustrating the input format
└── cascade/
    ├── __init__.py
    ├── data.py                 # JSON and JSONL reading/writing
    ├── models.py               # Embedding, reranker, and chat clients
    ├── rrf.py                  # Sentence windows + BM25/embedding fusion
    ├── rerank.py               # Reranking + evidence selection/merging
    ├── split.py                # Population and calibration sample
    ├── inference.py            # Batched evidence / shared full-document calls
    ├── ns.py                   # Two embedding views, NS scores, threshold grids
    └── wsr.py                  # Certification, routing, final results
```

Run every command below from the repository root. Each stage reads explicit input files and writes explicit outputs. Generated data and results go into `data/` and `runs/`.

## Environment and models

Use Python 3.10+ and a CUDA-enabled PyTorch installation for the local embedding and reranker models. The SLM serving command below uses vLLM on Linux or WSL2. Select GPU memory and context length for your models; the paper uses an A100 80 GB.

```bash
git clone https://github.com/oceanusXXD/cascade.git
cd cascade
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

The local model defaults in `config.json` are:

| Task | Model / interface |
| --- | --- |
| Retrieval and NS embeddings | `Qwen/Qwen3-Embedding-4B`, Sentence Transformers |
| Reranking | `Qwen/Qwen3-Reranker-0.6B`, Sentence Transformers CrossEncoder |
| SLM | `Qwen/Qwen3-4B-Instruct-2507`, vLLM chat endpoint |
| Oracle | Your full-document model, chat-completions-compatible endpoint |

Before the Oracle steps, edit `oracle.model`, `oracle.base_url`, and its prices, then set the key:

```bash
export ORACLE_API_KEY="your-key"
```

`oracle.request` contains provider generation options. Configure deterministic decoding and disable thinking for your Oracle. The client requests a single `Y`/`N` answer with a 16-token completion budget. Both chat endpoints must support `max_completion_tokens`; the SLM additionally needs token log probabilities and vLLM's `logprob_token_ids` option.

The embedding and reranker commands load their models directly and release them when the command exits. Prepare retrieval and both NS embedding views **before starting the SLM server**. NS scoring then runs on CPU, allowing one GPU to serve the SLM throughout both inference stages.

The interfaces follow the official [Qwen embedding](https://huggingface.co/Qwen/Qwen3-Embedding-4B), [Qwen reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), and [vLLM log-probability](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/) documentation. Use a vLLM version exposing `logprob_token_ids`.

## Input data

Prepare `data/records.jsonl`: one query–document pair per line.

```json
{"id":"q1-d1","query":"Does this review complain about the wait time?","document":"We waited an hour before being seated. The food was good.","label":"yes"}
{"id":"q1-d2","query":"Does this review complain about the wait time?","document":"The staff seated us immediately. Dinner was excellent.","label":"no"}
```

| Field | Meaning |
| --- | --- |
| `id` | Unique, nonempty string identifying the query–document pair |
| `query` | The complete predicate, including any claim, function name, or specification |
| `document` | The complete source text |
| `label` | Optional benchmark ground truth: `yes` or `no`; used only for final evaluation |

IDs can be source IDs or sequential strings. Repeated documents with different queries have different pair IDs. Each document must contain text. Supply benchmark-specific predicates directly in `query`.

`examples/records.jsonl` illustrates this format. The repository contains the core pipeline and these three example records; the six paper datasets and measured result tables are separate inputs. To reproduce a paper setting, prepare its actual query–document population and ground-truth labels in this format.

## Run the pipeline

### 1. Retrieve evidence candidates with RRF

```bash
python -m cascade.rrf \
  --input data/records.jsonl \
  --output runs/rrf.jsonl
```

For each record, BlingFire splits its document into sentences. Three-sentence windows with stride 1 are ranked **within that document** by BM25 and cosine embedding similarity. RRF combines their ranks with `k=60`; the largest score gap plus a five-window buffer determines the candidates sent to the reranker.

Output: the original record, sentences, and candidate windows with BM25, embedding, and RRF scores.

### 2. Rerank and merge evidence

```bash
python -m cascade.rerank \
  --input runs/rrf.jsonl \
  --output runs/evidence.jsonl
```

The reranker scores the surviving windows. Keep the prefix ending at the largest reranker-score gap, cap it at five windows, and merge their sentences in document order with overlapping sentence indices removed. With tied scores, retain the available prefix up to the cap.

Output: records with `evidence`, `selected_sentence_indices`, and `selected_chunks`. The full `document` stays available for later stages.

### 3. Draw the calibration sample

```bash
python -m cascade.split \
  --input runs/evidence.jsonl \
  --output-dir data/split \
  --seed 42
```

The default sample size is `min(N, max(500, ceil(0.01*N)))`. Set `--size 1000` to choose a fixed size before calibration. Sampling is uniform without replacement, and the calibration file preserves the random draw order.

Outputs:

- `data/split/population.jsonl`: all N records, the population being certified.
- `data/split/calibration.jsonl`: the ordered calibration sample, still part of that population.
- `data/split/remaining.jsonl`: the complement of the calibration sample.
- `data/split/split.json`: sample size and seed.

### 4. Prepare the two NS embedding views

```bash
python -m cascade.ns embed \
  --input data/split/population.jsonl \
  --evidence-output runs/evidence_vectors.npz \
  --document-output runs/document_vectors.npz
```

Stage 1 uses embeddings of query + evidence; Stage 2 uses query + full document. Each NPZ contains record IDs and normalized vectors. Stage 2 later selects vectors for its own maximal pool. Preparing these vectors needs no predictions or Oracle labels.

### 5. Start the SLM server

In a second terminal, install vLLM in its serving environment and keep this server running through Step 7:

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

The served name and URL must match `slm.model` and `slm.base_url`. `slm.tokenizer` must match the served model. The default localhost endpoint uses no key; set `SLM_API_KEY` if your server requires one.

### 6. Stage 1: batched evidence inference and NS

```bash
python -m cascade.inference stage1 \
  --input data/split/population.jsonl \
  --output runs/stage1.jsonl

python -m cascade.ns score \
  --stage stage1 \
  --input data/split/population.jsonl \
  --predictions runs/stage1.jsonl \
  --vectors runs/evidence_vectors.npz \
  --output runs/stage1_scores.jsonl \
  --grid runs/grid1.json \
  --stage2-input runs/stage2_input.jsonl
```

Stage 1 packs multiple records into **one prompt** with explicit DocID delimiters, up to an 8,192-token input budget. It reads both label log probabilities at each output label. A malformed output, missing paired log probabilities, or an oversized evidence record causes deferral.

For a valid prediction, base confidence is `sigmoid(abs(log p(Y) - log p(N)) / 15)`. NS uses the 20 nearest cosine neighbors, excludes the record itself, and weights agreeing predictions by nonnegative similarity. Support shrinks toward 0.5 using the median neighborhood mass, then adjusts the smoothed log odds:

```text
rho_i       = W_i / (W_i + median(W))
shrunk_NS_i = rho_i * NS_i + (1 - rho_i) * 0.5
score_i     = log((b_i + 0.001) / (1 - b_i + 0.001)) + gamma * (2 * shrunk_NS_i - 1)
```

Defaults use `gamma=1`. Missing predictions contribute no neighbor voting weight. Exact neighbor search uses FAISS in blocks; its work grows quadratically with population size.

`grid1.json` freezes up to 32 unique score quantiles. `stage2_input.jsonl` contains the maximal Stage-2 pool: records with a score below `max(grid1)`, plus invalid Stage-1 records.

### 7. Stage 2: full-document SLM inference and NS

```bash
python -m cascade.inference full \
  --model slm \
  --input runs/stage2_input.jsonl \
  --output runs/stage2.jsonl

python -m cascade.ns score \
  --stage stage2 \
  --input runs/stage2_input.jsonl \
  --predictions runs/stage2.jsonl \
  --vectors runs/document_vectors.npz \
  --output runs/stage2_scores.jsonl \
  --grid runs/grid2.json
```

Each record receives one full-document call. Stage-2 NS uses neighbors **within the fixed maximal Stage-2 pool**, and `grid2.json` freezes up to 64 unique score quantiles. Full documents exceeding the configured SLM context raise an error; select an appropriate model/context before running the population.

The SLM server can now be stopped. Both score files and grids are fixed before any calibration label is requested.

### 8. Label the calibration sample with the Oracle

```bash
python -m cascade.inference full \
  --model oracle \
  --input data/split/calibration.jsonl \
  --output runs/oracle_calibration.jsonl
```

This uses the same `full_inference` function and query/full-document prompt as Stage 2. The Oracle returns binary labels; SLM confidence extraction is disabled. Benchmark `label` fields stay out of certification.

### 9. Joint threshold search with WS&R

```bash
python -m cascade.wsr fit \
  --population data/split/population.jsonl \
  --stage1 runs/stage1_scores.jsonl \
  --stage2 runs/stage2_scores.jsonl \
  --grid1 runs/grid1.json \
  --grid2 runs/grid2.json \
  --calibration data/split/calibration.jsonl \
  --oracle-labels runs/oracle_calibration.jsonl \
  --output-dir runs/search
```

For each λ₁, scan λ₂ from largest to smallest. Each candidate's calibration loss is 1 only when an accepted SLM prediction disagrees with the Oracle. Oracle fallback contributes zero loss. WS&R evaluates betting wealth at `epsilon = 1 - tau`, centered on the residual population mean, with predictable aGRAPA bets capped at `0.5/(1-q_t)`.

Use the level `delta / number_of_lambda1_tracks` at every candidate in a track. Stop that track at its first failed certificate. Select the cheapest pair among all certified prefixes; an empty certified set selects full Oracle fallback.

Outputs:

- `runs/search/policy.json`: selected thresholds, target, per-track level, costs, and fallback status.
- `runs/search/candidates.jsonl`: candidates actually visited by the fixed-sequence scans.

### 10. Route records, run the residual Oracle calls, and merge

```bash
python -m cascade.wsr route \
  --population data/split/population.jsonl \
  --stage1 runs/stage1_scores.jsonl \
  --stage2 runs/stage2_scores.jsonl \
  --policy runs/search/policy.json \
  --oracle-labels runs/oracle_calibration.jsonl \
  --output-dir runs/route

python -m cascade.inference full \
  --model oracle \
  --input runs/route/pending_oracle.jsonl \
  --output runs/oracle_residual.jsonl

python -m cascade.wsr merge \
  --population data/split/population.jsonl \
  --accepted runs/route/accepted.jsonl \
  --oracle runs/oracle_residual.jsonl \
  --output-dir runs/final
```

Routing accepts at the first qualifying SLM stage. When a deferred record already has a calibration Oracle label, that label is reused. An empty residual file produces an empty Oracle output without contacting a model.

`runs/final/predictions.jsonl` contains one result per population record, in original order:

```json
{"id":"q1-d1","prediction":"yes","stage":"stage1"}
```

`runs/final/summary.json` reports stage counts. If every input record has a benchmark `label`, it also reports accuracy and binary macro-F1 against those labels.

## Parameters and cost interpretation

Model, NS, and `wsr fit` commands read `config.json`; use `--config path/to/config.json` to select another configuration. The `split` command takes its sample size and seed directly; `wsr route` and `wsr merge` read their supplied files.

| Parameter | Default |
| --- | --- |
| Sentence window / stride | 3 / 1 |
| RRF constant / recall buffer | 60 / 5 |
| Maximum selected windows | 5 |
| Stage-1 input token budget | 8,192 |
| SLM context window | 32,768 |
| Confidence temperature | 15 |
| NS neighbors / adjustment weight | 20 / 1 |
| Stage-1 / Stage-2 grid points | Up to 32 / 64 |
| Accuracy target τ / failure budget δ | 0.9 / 0.05 |
| Calibration sample | `min(N, max(500, ceil(0.01*N)))` |

Prices are per million tokens. Candidate selection minimizes estimated deployment cost: the Stage-1 bill, full-document SLM costs for records deferred by Stage 1, and estimated Oracle costs for the final residual. SLM token counts come from endpoint usage; Stage-1 batch usage is divided equally among its records. Oracle input lengths use the SLM tokenizer and output lengths use the preselected `oracle.expected_output_tokens`, so this cost is an estimate. Update the example prices for your actual models.

`policy.json` separately reports the maximal Stage-2 search pass and calibration-labeling costs. These candidate-independent terms preserve the selected thresholds. Retrieval/indexing compute and the final provider bill are outside this estimate.

Certification concerns agreement with the fixed Oracle on this specific finite population. Benchmark accuracy and macro-F1 are evaluated separately against `label`. Fix the models, generation settings, NS parameters, grids, sample size, τ, and δ before observing calibration labels; reuse stored Oracle decisions for the calibrated population.
