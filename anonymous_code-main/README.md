# CODE

Standalone runnable package for the cascade pipeline and retrieval stages.

## Scope

This directory is intentionally self-contained. It keeps only the code needed
to run the packaged command-line entry points:

- `algorithm/`: core `prepare -> step1 -> step2 -> step3` pipeline.
- `script/retrieval/`: BM25, embedding, RRF, rerank, and chunk selection.
- `script/`: shared data loading, schemas, labels, model calls, vectors, and IO.
- `requirements.txt`: lightweight runtime dependencies for local checks and
  chat/embedding-compatible endpoints.

Dataset paths are supplied explicitly through CLI arguments. Runtime outputs
are written under the selected `--artifacts-root`.

## Environment

Use Python 3.10 or newer. Run commands from `CODE/` unless noted otherwise.

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

The default requirements cover local smoke checks, BM25, and external
chat or embedding services. Local GPU execution is optional; install the
runtime-specific packages separately in the target CUDA/PyTorch environment
before using local model paths or local rerank backends.

## Data Contract

All entry points consume explicit file paths:

- `--queries`: JSON list or object containing query rows.
- `--documents`: JSON list or object containing document rows.
- `--ground-truth`: optional JSON labels for evaluation and label-backed Step3.

Supported row fields are normalized internally. The stable minimal schema is:

```json
{"query_id": "q1", "text": "predicate text"}
{"document_id": "d1", "text": "document text"}
{"query_id": "q1", "document_id": "d1", "label": "yes"}
```

Rows may also include precomputed Stage2 fields:

```json
{"stage2_prediction": "yes", "stage2_routing_score": 0.83}
```

## Verify

These checks do not require any external model service:

```bash
python -m algorithm.cli --help
python -m algorithm.cli prepare --help
python -m script.retrieval.cli --help
```

## Local BM25 Example

BM25 runs fully locally on the provided query/document files.

```bash
rm -rf output/local_bm25
PYTHONPATH=. python -m script.retrieval.cli bm25 \
  --queries <queries.json> \
  --documents <documents.json> \
  --ground-truth <labels.json> \
  --artifacts-root output/local_bm25 \
  --window-size 2
```

Expected files:

- `output/local_bm25/bm25_candidates.jsonl`
- `output/local_bm25/bm25_candidates.meta.json`
- `output/local_bm25/summary.json`

## Retrieval Stages

The retrieval CLI is stage-oriented:

```text
bm25 -> embedding -> rrf -> rerank -> selectchunk
```

- `bm25` reads the dataset files directly.
- `embedding` reads BM25 candidates and calls an embedding endpoint, unless the
  model argument is a local model path with a compatible local runtime installed.
- `rrf` combines ranked candidates.
- `rerank` calls a `/score`-compatible endpoint, or a compatible local
  generation reranker when a local backend URL is used.
- `selectchunk` converts rerank rows into selected chunk artifacts.

## Core Pipeline

The core pipeline writes all stage artifacts under one `--artifacts-root`.

### 1. Prepare

`prepare` builds `evidence_rows.jsonl`, Step1 batch inputs, and stage-specific
pair embeddings. Stage1 embeddings use the Step1 input text; Stage2 embeddings
use the Stage2 input text. It requires an embedding backend.

```bash
PYTHONPATH=. python -m algorithm.cli prepare \
  --mode document \
  --queries <queries.json> \
  --documents <documents.json> \
  --ground-truth <labels.json> \
  --artifacts-root output/core_run \
  --embedding-model <embedding-model> \
  --embedding-base-url <embedding-base-url>
```

### 2. Step1

`step1` reads prepared artifacts and requires a chat backend.

```bash
PYTHONPATH=. python -m algorithm.cli step1 \
  --mode document \
  --queries <queries.json> \
  --documents <documents.json> \
  --ground-truth <labels.json> \
  --artifacts-root output/core_run \
  --embedding-model <embedding-model> \
  --step1-chat-model <chat-model> \
  --step1-chat-base-url <chat-base-url>
```

### 3. Step2

`step2` reads prepared artifacts plus cached Step1 rows and fits the WS&R
decision policy with Bonferroni `lambda1` tracks, fixed-sequence `lambda2`
certification, and certified-set joint search. If a Stage2 chat backend is
provided, Step2 first runs full-document Stage2 inference for rows missing
`stage2_prediction`; otherwise it consumes the prepared
`stage2_prediction` and `stage2_routing_score` fields. Rows without a valid
Stage2 prediction are deferred rather than accepted by Stage2.

```bash
PYTHONPATH=. python -m algorithm.cli step2 \
  --mode document \
  --queries <queries.json> \
  --documents <documents.json> \
  --ground-truth <labels.json> \
  --artifacts-root output/core_run \
  --embedding-model <embedding-model> \
  --step2-chat-model <chat-model> \
  --step2-chat-base-url <chat-base-url> \
  --tau 0.95 \
  --delta-w 0.05 \
  --combiner-source source2
```

Omit `--step2-chat-model` and `--step2-chat-base-url` for no-GPU validation
against prepared Stage2 fields. `--combiner-source source1` uses the default
pre-registered combiner constants; `source2` searches a pre-registered theta
grid jointly with `lambda1` and `lambda2`.

### 4. Step3

`step3` reads cached Step2 rows. With `--ground-truth`, deferred rows can be
completed from labels without a chat backend. Without labels, provide a Step3
chat backend.

```bash
PYTHONPATH=. python -m algorithm.cli step3 \
  --mode document \
  --queries <queries.json> \
  --documents <documents.json> \
  --ground-truth <labels.json> \
  --artifacts-root output/core_run \
  --embedding-model <embedding-model>
```

Model-backed Step3:

```bash
PYTHONPATH=. python -m algorithm.cli step3 \
  --mode document \
  --queries <queries.json> \
  --documents <documents.json> \
  --artifacts-root output/core_run \
  --embedding-model <embedding-model> \
  --step3-chat-model <chat-model> \
  --step3-chat-base-url <chat-base-url>
```
