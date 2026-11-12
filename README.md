# CascadeFilter

This is the code repository for the paper **CascadeFilter: Cost-Efficient Semantic Filtering with Joint Threshold Selection**.

**[Technical report (PDF)](CascadeFilter-technical-report.pdf)** contains the proofs and supplementary details.

```text
cascade/
├── code/                               # Implementation, requirements and example
├── CascadeFilter-technical-report.pdf  # Technical report
└── README.md
```

CascadeFilter performs three-stage semantic filtering. Each query–document pair receives a binary prediction and the stage that produced it.

| Stage | View | Execution |
| --- | --- | --- |
| 1 | Retrieved and reranked evidence | Several records in one SLM prompt, with individual DocIDs |
| 2 | Full document | One SLM call per record in the fixed maximal Stage-2 pool |
| 3 | Full document | Fixed Oracle label for records deferred by both SLM stages |

Prepare predictions, routing scores, quantile grids, and candidate costs **before drawing calibration records**. Draw an ordered uniform sample without replacement, obtain its Oracle labels, certify complete threshold pairs, and select the cheapest certified pair. An empty certified set uses all-Oracle fallback.

## Requirements and paper defaults

Use Linux or WSL2, Python 3.10+, and CUDA-enabled PyTorch. The paper uses one A100-80GB. Commands below use Bash and, after the initial setup, run from the `code/` directory.

```bash
git clone https://github.com/oceanusXXD/cascade.git
cd cascade/code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
cp examples/records.jsonl data/records.jsonl
```

| Setting | Default / paper setting |
| --- | --- |
| SLM | `Qwen/Qwen3-4B-Instruct-2507` |
| Embeddings | `Qwen/Qwen3-Embedding-4B` |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` |
| Oracle in the paper | GPT-5.5, thinking disabled |
| Sentence windows / stride | 3 / 1 |
| RRF constant / extra candidates | 60 / 5 |
| Evidence cap | 5 chunks |
| Stage-1 input budget | 8,192 tokens, including instructions and chat template |
| Generation temperature / score temperature | 0 / 15 |
| NS neighbors / adjustment weight / log-odds smoothing | Up to 20 / 1 / 0.001 |
| Shrinkage scale | Median neighborhood mass in the stage pool |
| Stage-1 / Stage-2 quantile levels | 20 / 32, including endpoints; duplicate values merged |
| Score families | 1 |
| Target Oracle agreement / failure probability | 0.9 / 0.05 |
| Calibration size | Rounded 1% of N, raised to the aGRAPA zero-error floor n0, capped at N |
| Calibration draws per experimental setting | 20 |
| Nominal SLM input / output USD per million tokens | 0.2 / 0.2 |
| Nominal Oracle input / output USD per million tokens | 5 / 30 |

`sentence-transformers>=5.4.0` supplies Qwen3 causal reranker support through `CrossEncoder`; its sigmoid scores are normalized yes/no relevance probabilities. See the [official model instructions](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#usage). The SLM endpoint must support vLLM's `logprob_token_ids`. The client requests the same `top_logprobs` count as the explicit token list, as required by the [vLLM sampling interface](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/).

## Input

UTF-8 JSONL, one query–document pair per line:

```json
{"id":"review-1","query":"Does this review complain about the wait time?","document":"We waited an hour before being seated.","label":"yes"}
{"id":"review-2","query":"Does this review complain about the wait time?","document":"The staff seated us immediately.","label":"no"}
```

Use unique nonempty string IDs. `query` contains the complete predicate; `document` contains the full source text. Give distinct IDs to a document evaluated against different predicates. Optional `label` is benchmark ground truth (`yes` / `no`), supplied for every record or omitted throughout. It is used only for final Accuracy and Macro-F1. Calibration uses Oracle `prediction` values.

The three supplied records demonstrate the input format. Replace them with the population for your setting.

## Prepare evidence and embeddings

Run these commands before starting the SLM server on the same GPU. Each local model exits and releases its memory after its command.

```bash
python rrf.py --input data/records.jsonl --output runs/rrf.jsonl \
  --embedding-model Qwen/Qwen3-Embedding-4B --device cuda \
  --window-size 3 --rrf-k 60 --buffer 5

python rerank.py --input runs/rrf.jsonl --output runs/evidence.jsonl \
  --reranker-model Qwen/Qwen3-Reranker-0.6B --device cuda --max-chunks 5

python ns.py embed --input runs/evidence.jsonl \
  --evidence-output runs/evidence_vectors.npz \
  --document-output runs/document_vectors.npz \
  --embedding-model Qwen/Qwen3-Embedding-4B --device cuda
```

RRF combines BM25 and dense ranks within each document. The first cutoff keeps the prefix before the largest RRF-score gap, plus five windows. Reranking applies a second largest-gap cutoff with a five-chunk cap. Selected sentences are deduplicated by index and merged in document order. Tied scores retain the available prefix, subject to the cap.

The NPZ files contain IDs and normalized vectors for query + evidence and query + full document. Stage-2 NS later uses only vectors in its maximal pool. Local embedding and reranking inputs use their configured `--max-length`; full-document SLM inference raises an error if a document exceeds its context, so choose the required context before preparing the population.

## Prepare the two SLM stages

Start vLLM in a **second terminal**, from the `cascade/code/` directory, with a separate environment:

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -U vllm
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 --max-model-len 32768 --generation-config vllm
```

Wait for the server to be ready. Continue in the first terminal using `.venv`:

```bash
python inference.py stage1 --input runs/evidence.jsonl --output runs/stage1.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 --base-url http://localhost:8000/v1 \
  --temperature 0 --seed 42 --batch-token-budget 8192 \
  --context-window 32768 --score-temperature 15

python ns.py score --stage stage1 --input runs/evidence.jsonl \
  --predictions runs/stage1.jsonl --vectors runs/evidence_vectors.npz \
  --output runs/stage1_scores.jsonl --grid runs/grid1.json \
  --stage2-input runs/stage2_input.jsonl --k 20 --gamma 1 --grid-size 20

python inference.py full --input runs/stage2_input.jsonl --output runs/stage2.jsonl \
  --model Qwen/Qwen3-4B-Instruct-2507 --base-url http://localhost:8000/v1 \
  --temperature 0 --seed 42 --context-window 32768 --score-temperature 15

python ns.py score --stage stage2 --input runs/stage2_input.jsonl \
  --predictions runs/stage2.jsonl --vectors runs/document_vectors.npz \
  --output runs/stage2_scores.jsonl --grid runs/grid2.json \
  --k 20 --gamma 1 --grid-size 32
```

The tokenizer defaults to `--model`; pass `--tokenizer` when the endpoint uses a different served alias. For an authenticated local endpoint, pass `--api-key-env SLM_API_KEY` and set that environment variable.

Stage 1 emits `DocID: Y` / `DocID: N` lines. Each unique valid DocID retains its result; malformed, missing, duplicated, or unscorable entries defer their corresponding records. Missing paired Y/N probabilities also causes deferral. Stage 2 emits one Y/N label per call.

The maximal Stage-2 pool is every record with `score1 < max(grid1)`, including invalid Stage-1 records, whose scores act as negative infinity. Stage-2 predictions and NS remain fixed across all candidate thresholds. An empty pool uses a singleton dummy second-stage grid.

Stop the SLM server after both stages finish.

## Freeze costs and draw calibration records

Use unchanged population and score files throughout this run:

```bash
python wsr.py prepare --population runs/evidence.jsonl \
  --stage1 runs/stage1_scores.jsonl --stage2 runs/stage2_scores.jsonl \
  --grid1 runs/grid1.json --grid2 runs/grid2.json \
  --slm-input-price 0.2 --slm-output-price 0.2 \
  --oracle-input-price 5 --oracle-output-price 30 --oracle-output-tokens 64 \
  --output runs/prepared.json

python split.py --input runs/evidence.jsonl --prepared runs/prepared.json \
  --output-dir data/split --tau 0.9 --delta 0.05 --seed 42
```

`prepared.json` freezes up to 640 candidate costs and routing counts. Estimated Oracle input lengths default to the SLM tokenizer's full-prompt count. To use provider-specific input lengths, pass `--oracle-inputs data/oracle_input_tokens.jsonl` with `{"id":"review-1","input_tokens":123}` for every record. These counts must be prepared from prompts without Oracle labels before sampling. The 64-token output estimate is fixed separately from the 128-token Oracle completion limit.

`split.py` writes `calibration.jsonl` in random draw order and `split.json` with the chosen budget and seed. The population continues to live at `runs/evidence.jsonl`; calibration records remain members of it.

The automatic budget is `min(N, max((N + 50) // 100, n0))`. The shared aGRAPA implementation computes n0 from zero losses before drawing S. If its boundary is unreachable within N steps, calibration uses the full population. A fixed override is available as `--size`. Freeze the budget, seed, tau, delta, models, scoring settings and costs before viewing the sample or its labels. Keep calibration file order unchanged.

## Obtain labels, select thresholds, and execute

Set the Oracle endpoint, served model and credential. Match the paper by using its GPT-5.5 Oracle in a supported non-thinking mode; other model choices define a different experimental configuration.

```bash
export ORACLE_MODEL="gpt-5.5"
export ORACLE_BASE_URL="https://api.openai.com/v1"
export ORACLE_API_KEY="your-key"

python inference.py full --oracle --input data/split/calibration.jsonl \
  --output runs/oracle_calibration.jsonl --model "$ORACLE_MODEL" \
  --base-url "$ORACLE_BASE_URL" --temperature 0 \
  --reasoning-effort none --max-output-tokens 128

python wsr.py fit --population runs/evidence.jsonl \
  --stage1 runs/stage1_scores.jsonl --stage2 runs/stage2_scores.jsonl \
  --prepared runs/prepared.json --calibration data/split/calibration.jsonl \
  --oracle-labels runs/oracle_calibration.jsonl --tau 0.9 --delta 0.05 \
  --output-dir runs/search

python wsr.py route --population runs/evidence.jsonl \
  --stage1 runs/stage1_scores.jsonl --stage2 runs/stage2_scores.jsonl \
  --policy runs/search/policy.json --oracle-labels runs/oracle_calibration.jsonl \
  --output-dir runs/route

python inference.py full --oracle --input runs/route/pending_oracle.jsonl \
  --output runs/oracle_residual.jsonl --model "$ORACLE_MODEL" \
  --base-url "$ORACLE_BASE_URL" --temperature 0 \
  --reasoning-effort none --max-output-tokens 128

python wsr.py merge --population runs/evidence.jsonl \
  --accepted runs/route/accepted.jsonl --oracle runs/oracle_residual.jsonl \
  --oracle-labels runs/oracle_calibration.jsonl --policy runs/search/policy.json \
  --output-dir runs/final
```

Both chat endpoints must accept `max_completion_tokens`. Adapt the non-thinking option for your provider (`--disable-thinking` for a supported vLLM chat template), using the same model version, prompt, endpoint and decoding settings for calibration and fallback. Oracle output is Y/N on the first line and a short justification on the second. Calibration labels are reused automatically for fallback. Empty residual inputs write an empty result file without an API call.

| Output | Contents |
| --- | --- |
| `runs/search/policy.json` | Selected certified pair or `selected: null`, calibration labels, grids, cost assumptions |
| `runs/search/candidates.jsonl` | Visited candidates, terminal log wealth, decisions, routing counts and frozen plan costs |
| `runs/route/accepted.jsonl` | SLM results and reused calibration Oracle fallback labels |
| `runs/route/pending_oracle.jsonl` | Remaining full-document Oracle inputs |
| `runs/final/predictions.jsonl` | One prediction and stage per record, in population order |
| `runs/final/summary.json` | Routing rates, benchmark metrics when labels exist, Oracle usage and costs |

Append `--prep-cost AMOUNT` to `merge` to include measured retrieval, embedding, reranking, indexing and search compute in USD for the population. Without it, `prepared_batch_total` stays null and `prepared_batch_inference` reports model inference alone.

Append `--oracle-reference data/oracle_reference.jsonl` to `merge` to measure full-population `oracle_agreement` and `oracle_violation` (`agreement < tau`). The reference must contain the same fixed Oracle labels used for calibration and fallback. Benchmark `label` values cannot substitute for this reference.

## Twenty independent calibration draws

For the paper's repeated-draw diagnostics, fix one full-population Oracle label map before the 20 draws, alongside the prepared SLM outputs. `inference.py full --oracle --cache FILE` reuses existing labels by ID and queries only missing records; a complete cache needs no API call or credential. Keep IDs tied to the same population and model configuration. For example, materialize the reference before the repeated-draw experiment:

```bash
python inference.py full --oracle --input runs/evidence.jsonl \
  --cache runs/oracle_calibration.jsonl --output data/oracle_reference.jsonl \
  --model "$ORACLE_MODEL" --base-url "$ORACLE_BASE_URL" \
  --temperature 0 --reasoning-effort none --max-output-tokens 128
```

This full labeling supports offline evaluation; its full-population annotation expense is separate from a deployment run's calibration and fallback costs. Each draw exposes only its sampled labels to fitting:

```bash
set -euo pipefail
for seed in {1..20}; do
  draw="runs/draws/$seed"
  python split.py --input runs/evidence.jsonl --prepared runs/prepared.json \
    --output-dir "$draw/split" --tau 0.9 --delta 0.05 --seed "$seed"
  python inference.py full --oracle --input "$draw/split/calibration.jsonl" \
    --cache data/oracle_reference.jsonl --output "$draw/oracle_calibration.jsonl" \
    --model "$ORACLE_MODEL" --base-url "$ORACLE_BASE_URL"
  python wsr.py fit --population runs/evidence.jsonl \
    --stage1 runs/stage1_scores.jsonl --stage2 runs/stage2_scores.jsonl \
    --prepared runs/prepared.json --calibration "$draw/split/calibration.jsonl" \
    --oracle-labels "$draw/oracle_calibration.jsonl" --tau 0.9 --delta 0.05 \
    --output-dir "$draw/search"
  python wsr.py route --population runs/evidence.jsonl \
    --stage1 runs/stage1_scores.jsonl --stage2 runs/stage2_scores.jsonl \
    --policy "$draw/search/policy.json" --oracle-labels "$draw/oracle_calibration.jsonl" \
    --output-dir "$draw/route"
  python inference.py full --oracle --input "$draw/route/pending_oracle.jsonl" \
    --cache data/oracle_reference.jsonl --output "$draw/oracle_residual.jsonl" \
    --model "$ORACLE_MODEL" --base-url "$ORACLE_BASE_URL"
  python wsr.py merge --population runs/evidence.jsonl \
    --accepted "$draw/route/accepted.jsonl" --oracle "$draw/oracle_residual.jsonl" \
    --oracle-labels "$draw/oracle_calibration.jsonl" --policy "$draw/search/policy.json" \
    --oracle-reference data/oracle_reference.jsonl --output-dir "$draw/final"
done

python wsr.py summarize --summaries runs/draws/*/final/summary.json \
  --output runs/draws/summary.json
```

Choose all seeds in advance and retain every draw, including all-Oracle fallback. The aggregate reports arithmetic means and **sample** standard deviations (`ddof=1`) plus violation counts. Each certificate concerns one draw; joint validity across deployments requires an additional failure-budget allocation. Cached replay costs charge the recorded token usage for each member of S union R once per simulated run; replay itself sends no new request.

## Scores, certification and cost definitions

Both SLM stages use `inference.py:label_score`. The sign of the Y/N log-probability margin determines the prediction; its magnitude determines the base confidence:

```text
margin     = log p(Y) - log p(N)
prediction = yes if margin > 0, otherwise no
b          = sigmoid(abs(margin) / T), T = 15 by default
w_ij       = max(cosine(v_i, v_j), 0), excluding self
NS_i       = similarity-weighted fraction of matching neighbor predictions
W_i        = sum_j w_ij
rho_i      = W_i / (W_i + median(W))
shrunk_NS  = rho_i * NS_i + (1 - rho_i) / 2
score_i    = log((b_i + 0.001) / (1 - b_i + 0.001)) + gamma * (2 * shrunk_NS - 1)
```

Base confidence is in `[0.5, 1]`: stronger Y and stronger N preferences both increase it. A zero margin gives confidence 0.5 and predicts `no`, following the strict `margin > 0` prediction rule. `--score-temperature` controls T independently of generation temperature, which stays at zero.

Zero mass gives neutral support; a zero shrinkage denominator sets rho to zero. Invalid model predictions receive no routing score. As a fallback convention, an invalid neighbor contributes its weight to mass and no matching vote. Stage 1 uses the whole population; Stage 2 uses its frozen maximal pool. FAISS searches exact cosine neighbors in blocks; neighbor search therefore grows quadratically with pool size.

For each candidate, calibration loss is one exactly when an accepted SLM prediction disagrees with the Oracle. Fallback has zero Oracle-relative loss. `betting.py` implements the residual-population q, predictable aGRAPA bet and log-wealth updates from the appendix. Calibration sizing and certification share that implementation.

For each fixed lambda1, `wsr.py fit` scans lambda2 from largest to smallest, stopping at the first failure. Every visited candidate uses level `delta / L1`, with terminal log wealth compared against `log(L1 / delta)`. Selection minimizes frozen plan cost among all certified pairs. The certificate targets agreement with the fixed Oracle over this finite population. Benchmark Accuracy and Macro-F1 are measured separately. A new population requires fresh calibration.

```text
selected-plan estimate C(phi)
  = Stage1(all) + Stage2(defer1) + estimated_Oracle(defer_both)

prepared-batch cost C_batch(phi; S)
  = preprocessing + Stage1(all) + Stage2(maximal_pool) + Oracle(S union residual)
```

The selected-plan objective excludes search preparation. The prepared-batch summary counts each cached stage invocation once and counts calibration plus uncached residual Oracle usage. It uses measured API token counts at the frozen nominal prices. Provider discounts and a complete billing ledger are outside that nominal measure.

Candidate `oracle_count` and `oracle_rate`, and final `stage_counts.oracle` and `stage_rates.oracle`, describe **fallback routing**. They include fallback records whose labels are already cached from calibration. The number of distinct Oracle evaluations for calibration and execution is `|S union residual| = n + |residual| - |S intersection residual|`, reported as `oracle_usage.unique_records`. It can differ from the fallback count.

## Notation and output fields

| Symbol | Meaning and code representation |
| --- | --- |
| $\mathcal I,N$; $q_i,d_i$ | Fixed population and its size; each record's `query` and `document` |
| $\mathcal M_S,\mathcal M_L$ | SLM and Oracle selected through `--model` |
| $y_i,\hat y_i^{(s)}$ | Fixed Oracle `prediction`; SLM `prediction` at Stage s |
| $\ell_i^{(s)},b_i^{(s)}$ | Signed `yes_logprob - no_logprob`; `base_confidence = sigmoid(abs(margin) / T)` |
| $NS_i^{(s)},W_i^{(s)},\widetilde{NS}_i^{(s)}$ | `neighbor_support`, `neighbor_mass`, `shrunk_support` |
| $\rho_i^{(s)},\sigma_0^2$ | Shrinkage weight `rho` and stage-pool median mass `sigma_sq` in `ns.py` |
| $\tilde z_i^{(s)}$ | Final routing `score`, with adjustment weight `--gamma` and smoothing `--epsilon` |
| $\lambda_s,\Lambda_s,L_s$ | Acceptance threshold, frozen `thresholds` grid, and its distinct size |
| $A_i^{(s)},D_i^{(s)}$ | Accept when `score >= lambda_s`; defer otherwise |
| $\mathcal I_2^{\max}$ | Fixed maximal Stage-2 pool in `stage2_input.jsonl` |
| $S,n,n_0$ | Ordered calibration sample, its size, and `zero_error_min_size` |
| $\tau,\varepsilon,\delta$ | Target Oracle agreement `--tau`, error target `1 - tau`, failure budget `--delta` |
| $\eta_f$ | Per-track testing level `level_per_track = delta / L1` |
| $M_n^+(\varepsilon;\phi)$ | Candidate's terminal betting wealth, stored as `log_wealth` in log space |
| $\phi,\mathcal C,\hat\phi$ | Threshold pair, certified candidate set, and cost-minimizing certified `selected` pair |
| $c_{si},C(\phi)$ | Fixed per-record stage cost and `estimated_plan_cost` |
| $R(\phi),r_3(\phi)$ | Fallback records and their population fraction, recorded as candidate `oracle_count` / `oracle_rate` |
| $r_{\mathrm{reach2}}(\phi)$ | Fraction reaching Stage 2, computed as `(N - stage1_count) / N` |
| $C_{\rm prep}$ | Preprocessing compute cost supplied through `merge --prep-cost` |
| $C_{\rm batch}(\phi;S)$ | Cached-batch total `costs_usd.prepared_batch_total`, including supplied preprocessing cost |

The `ns.py --epsilon` option is log-odds smoothing, corresponding to epsilon_log. The certification error target epsilon is always computed as `1 - tau`.

## Paper-to-code map

| Manuscript component | Implementation |
| --- | --- |
| Algorithm 1: preparation, selection, execution | `rrf.py`, `rerank.py`, `inference.py`, `ns.py`, `wsr.py` |
| Sentence windows, hybrid RRF and adaptive evidence cutoff | `rrf.py:retrieve`, `rerank.py:select_evidence` |
| Batched label-position margins and full-document inference | `inference.py:parse_slm`, `full_inference` |
| Pool-specific NS, median-mass shrinkage and quantile grids | `ns.py:score_rows` and `score` command |
| Ordered SRSWOR and zero-error calibration floor | `split.py:sample_calibration`, `betting.py:zero_error_min_size` |
| Algorithm 2: fixed sequence and track Bonferroni | `wsr.py:joint_search` |
| Cost objective and cache-aware batch accounting | `wsr.py:prepare_search`, `execution_summary` |
| Repeated-draw metrics and violations | `wsr.py` merge command and `summarize_runs` |

Supported score ablations: set `--gamma 0` on either or both stage scoring commands to remove NS; use `--no-shrinkage` for rho=1, or `--uniform-weights` for unit neighbor weights with the median mass recomputed. Rebuild scores, grids, maximal pool when affected, costs and calibration for each variant.

## Source layout

```text
cascade/
  code/
    requirements.txt
    common.py       # Shared JSONL/JSON I/O and embedding model
    betting.py      # Shared without-replacement aGRAPA updates
    rrf.py          # Sentence windows, BM25, dense retrieval, RRF
    rerank.py       # Qwen3 reranking and sentence merge
    inference.py    # SLM inference, label margins, Oracle cache reuse
    ns.py           # Stage-specific embeddings, NS scores, quantile grids
    split.py        # Ordered calibration sample and automatic budget
    wsr.py          # Preparation, certification, routing, metrics and costs
    examples/records.jsonl
  CascadeFilter-technical-report.pdf
  README.md
```

Generated data and results belong under `code/data/` and `code/runs/`. Rebuild downstream artifacts when changing a population, model, score rule, grid or price assumption. Keep frozen files unchanged during each calibration run.
