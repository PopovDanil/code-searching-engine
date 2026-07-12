# Query Rewriting and Language-Aware Reranking Experiments

## Goal

This experiment evaluates two query rewriting strategies and an explicit programming-language hint for the reranker prompt in the semantic code search pipeline.

The tested strategies are:

- baseline reranking without query rewriting or language hint;
- direct LLM query rewriting;
- HyDE-style hypothetical answer generation;
- language-aware reranking;
- HyDE combined with language-aware reranking.

## Evaluation setup

All experiments were run on Google Colab with CUDA.

| Parameter | Value |
|---|---|
| Dataset | CodeSearchNet |
| Language | Python |
| Requested dataset records | 500 |
| Loaded valid records | 498 |
| Indexed code chunks | 579 |
| Evaluation queries | 50 |
| Embedding model | `Qwen/Qwen3-Embedding-0.6B` |
| Reranker model | `Qwen/Qwen3-Reranker-0.6B` |
| Query rewriter model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Device | CUDA |
| Embedding dtype | `float16` |
| FAISS index type | `flat` |
| Retrieval candidates before reranking | 100 |
| Separate indexes | disabled |
| Max sequence length | 512 |
| Max chunk characters | 1500 |
| Chunk overlap characters | 150 |

The command format was:

```bash
python src/cli.py evaluate \
  --config configs/<config_name>.yaml \
  --languages python \
  --max-dataset-records 500 \
  --max-queries 50
```

## Results

| Experiment | Config | Reranker | Query rewriting | Language hint | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Baseline | `configs/colab_baseline.yaml` | yes | none | no | 0.5200 | 0.7600 | 0.7800 | 0.6208 | 0.6606 |
| Rewrite | `configs/colab_rewrite.yaml` | yes | rewrite | no | 0.3200 | 0.7600 | 0.7800 | 0.4850 | 0.5580 |
| HyDE | `configs/colab_hyde.yaml` | yes | hyde | no | 0.2000 | 0.5000 | 0.6800 | 0.3371 | 0.4180 |
| Language hint | `configs/colab_language_hint.yaml` | yes | none | yes | 0.6200 | 0.7800 | 0.8000 | 0.6857 | 0.7139 |
| HyDE + language hint | `configs/colab_hyde_language_hint.yaml` | yes | hyde | yes | 0.1800 | 0.5800 | 0.6800 | 0.3364 | 0.4194 |

## Comparison against baseline

| Experiment | Δ Recall@1 | Δ Recall@5 | Δ Recall@10 | Δ MRR | Δ NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Rewrite | -0.2000 | 0.0000 | 0.0000 | -0.1358 | -0.1026 |
| HyDE | -0.3200 | -0.2600 | -0.1000 | -0.2837 | -0.2426 |
| Language hint | +0.1000 | +0.0200 | +0.0200 | +0.0649 | +0.0533 |
| HyDE + language hint | -0.3400 | -0.1800 | -0.1000 | -0.2844 | -0.2412 |

## Interpretation

The explicit programming-language hint in the reranker prompt produced the best result. It improved all key metrics compared with the vanilla reranker baseline:

- Recall@1 improved from 0.5200 to 0.6200.
- Recall@5 improved from 0.7600 to 0.7800.
- Recall@10 improved from 0.7800 to 0.8000.
- MRR improved from 0.6208 to 0.6857.
- NDCG@10 improved from 0.6606 to 0.7139.

Direct query rewriting degraded ranking quality. Recall@5 and Recall@10 stayed the same as the baseline, but Recall@1, MRR, and NDCG@10 decreased. This suggests that the rewrite model sometimes changed the query too aggressively or removed useful information.

HyDE performed substantially worse. The generated hypothetical descriptions were often too long, generic, or contained hallucinated APIs and implementation details. This added noise to the retrieval query and reduced both retrieval and ranking quality.

HyDE combined with the language-aware reranker still performed poorly. The language hint helped the reranker in the clean-query setting, but it could not compensate for noisy HyDE-generated retrieval queries.

Overall, the best tested configuration was the language-aware reranker without query rewriting.

## Notes

The vanilla reranker did not outperform the earlier embedding-only local run in all metrics. For this reason, additional embedding-only configs were added:

- `configs/colab_embedding_only_100.yaml`
- `configs/colab_embedding_only_10.yaml`

These configs allow a clean ablation between embedding-only retrieval and reranker-based ranking under the same Colab setup.
