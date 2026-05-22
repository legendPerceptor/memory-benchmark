# Memory System Benchmark Tool

一套统一评测工具，用于测试多种 Memory 系统在对话长期记忆数据集上的表现。

## 功能特点

- **统一接口**: 通过适配器模式，支持多种 Memory 系统
- **多种指标**: 检索召回 (R@K, NDCG@K, MRR) + 问答准确率 (F1, EM)
- **性能监控**: 索引时间、查询延迟、内存占用
- **一键运行**: 通过命令行脚本快速运行基准测试
- **多格式报告**: 支持 JSON 和 Markdown 格式输出

## 支持的系统

| 系统 | 描述 | 接口方式 |
|------|------|----------|
| MemPalace | 逐字存储，AAAK 压缩 | Python API |
| oG-Memory | 6 阶段生命周期，L0/L1/L2 检索 | HTTP API |
| memsearch | Markdown 源，混合搜索 | Python API |

## 支持的数据集

| 数据集 | 描述 | QA 数量 |
|--------|------|---------|
| LoCoMo | ACL 2024 对话长期记忆 | ~2000 |

## 安装依赖

```bash
# 核心依赖
pip install chromadb psutil pyyaml

# 可选：fastembed 用于自定义嵌入模型
pip install fastembed

# 可选：memsearch 相关依赖
pip install pymilvus
```

## 快速开始

### 1. 克隆仓库

```bash
cd /home/yuanjian/Development/memory-projects/memory-systems
```

### 2. 运行基准测试

```bash
cd memory-benchmark

# 运行 MemPalace 基准测试
./scripts/benchmark.sh --system mempalace \
    --data /home/yuanjian/Development/memory-projects/memory-eval/locomo/data/locomo10.json

# 运行多系统对比
./scripts/benchmark.sh --systems mempalace,ogmemory,memsearch \
    --data /home/yuanjian/Development/memory-projects/memory-eval/locomo/data/locomo10.json

# 指定参数
./scripts/benchmark.sh --system mempalace \
    --top-k 10 \
    --mode hybrid \
    --granularity session
```

### 3. 查看报告

报告保存在 `reports/` 目录下：

```bash
ls reports/
# locomo_benchmark_20260521_173253.md
# locomo_benchmark_20260521_173253.json

# 查看 Markdown 报告
cat reports/locomo_benchmark_*.md
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--system` | 要测试的 Memory 系统 | mempalace |
| `--systems` | 逗号分隔的多个系统 | - |
| `--benchmark` | 基准测试名称 | locomo |
| `--data` | 数据集路径 | - |
| `--top-k` | 检索返回结果数量 | 10 |
| `--granularity` | 语料粒度 (session/dialog) | session |
| `--mode` | 检索模式 | raw |
| `--output` | 输出目录 | reports |
| `--format` | 输出格式 (json/markdown/both) | both |

## Python API

```python
from adapters.mempalace import MemPalaceAdapter
from benchmarks.locomo import LoCoMoBenchmark
from reporters.markdown import generate_markdown_report
from reporters.json import generate_json_report

# 创建适配器
adapter = MemPalaceAdapter(mode="raw", granularity="session")

# 创建基准测试
benchmark = LoCoMoBenchmark(
    data_path="/path/to/locomo10.json",
    adapter=adapter,
    granularity="session",
)

# 加载数据
benchmark.load_data("/path/to/locomo10.json")

# 准备语料并索引
corpus = benchmark.prepare_corpus()
adapter.index(corpus)

# 准备查询并评估
queries = benchmark.prepare_queries()
results = []
for query in queries:
    retrieved = adapter.search(query.question, top_k=10)
    result = benchmark.evaluate_query(query, 10, retrieved)
    results.append(result)

# 生成报告
generate_markdown_report(results, "report.md")
generate_json_report(results, "report.json")
```

## 项目结构

```
memory-benchmark/
├── config/
│   └── config.yaml           # 配置文件
├── src/
│   ├── adapters/             # Memory 系统适配器
│   │   ├── base.py          # 适配器基类
│   │   ├── mempalace.py     # MemPalace 适配器
│   │   ├── ogmemory.py      # oG-Memory 适配器
│   │   └── memsearch.py     # memsearch 适配器
│   ├── benchmarks/           # 基准测试模块
│   │   ├── base.py          # 基准测试基类
│   │   └── locomo.py        # LoCoMo 数据集适配器
│   ├── metrics/              # 评估指标
│   │   ├── retrieval.py     # R@K, NDCG@K, MRR
│   │   ├── qa.py           # F1, EM, ROUGE-L
│   │   └── performance.py # 性能指标
│   ├── reporters/            # 报告生成器
│   │   ├── json.py         # JSON 报告
│   │   └── markdown.py     # Markdown 报告
│   └── runner.py            # 统一入口
├── scripts/
│   └── benchmark.sh          # 一键运行脚本
├── reports/                   # 生成的报告
└── README.md
```

## 添加新的 Memory 系统

创建一个新的适配器，继承 `MemorySystemAdapter` 基类：

```python
from adapters.base import MemorySystemAdapter, RetrievalResult, IndexResult

class MyMemoryAdapter(MemorySystemAdapter):
    name = "mymemory"

    def health_check(self) -> bool:
        # 检查服务是否可用
        return True

    def index(self, corpus: list[dict]) -> IndexResult:
        # 索引语料
        indexed_count = len(corpus)
        return IndexResult(indexed_count=indexed_count, duration_ms=0)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        # 检索相关文档
        results = []
        # ... 检索逻辑
        return results

    def reset(self) -> None:
        # 清空索引
        pass
```

然后在 `runner.py` 中注册：

```python
from adapters.mymemory import MyMemoryAdapter

ADAPTER_REGISTRY = {
    "mymemory": MyMemoryAdapter,
    # ...
}
```

## 添加新的数据集

创建一个新的基准测试，继承 `Benchmark` 基类：

```python
from benchmarks.base import Benchmark, QueryEntry, EvaluationResult

class MyDatasetBenchmark(Benchmark):
    name = "mydataset"

    def prepare_corpus(self) -> list[dict]:
        # 准备要索引的语料
        pass

    def prepare_queries(self) -> list[QueryEntry]:
        # 准备查询
        pass

    def evaluate_query(self, query, top_k, retrieved) -> EvaluationResult:
        # 评估单个查询
        pass
```

## 评估指标

### 检索指标
- **Recall@K**: top-K 中命中的正确答案比例
- **NDCG@K**: 归一化折损累计增益
- **MRR**: 平均倒数排名

### 问答指标
- **F1 Score**: 词级别的 F1 分数
- **Exact Match**: 精确匹配率
- **ROUGE-L**: 最长公共子序列

### 性能指标
- 索引时间
- 查询延迟 (avg, median, P95, P99)
- 内存占用

## 许可证

MIT
