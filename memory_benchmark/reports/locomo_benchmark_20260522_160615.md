# Memory System Benchmark Report

**Generated**: 2026-05-22 16:12:59
**Benchmark**: LoCoMo
**Dataset**: LoCoMo (10 conversations, ~2000 QA pairs)

## Summary

- **Total Questions**: 1986
- **Systems Evaluated**: 1
- **Total Time**: 403.7s

## Retrieval Metrics Comparison

| System | R@1 | R@5 | R@10 | NDCG@10 | MRR |
|--------|-----|-----|------|---------|-----|
| mempalace (raw) | 0.214 | 0.423 | 0.423 | 0.000 | 0.321 |

## System Configurations

### mempalace

```yaml
  name: mempalace
  mode: raw
  embed_model: default
  collection_name: locomo_test
  granularity: session
```
