"""
Memory Benchmark Runner
统一入口，调度各模块运行基准测试
"""

import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .adapters.mempalace import MemPalaceAdapter
from .adapters.ogmemory import OGMemoryAdapter
from .adapters.memsearch import MemSearchAdapter
from .benchmarks.locomo import LoCoMoBenchmark
from .metrics.retrieval import recall_at_k, ndcg_at_k, mrr
from .metrics.qa import compute_qa_metrics
from .metrics.performance import PerformanceMonitor
from .reporters.json import generate_json_report
from .reporters.markdown import generate_markdown_report


ADAPTER_REGISTRY = {
    "mempalace": MemPalaceAdapter,
    "ogmemory": OGMemoryAdapter,
    "memsearch": MemSearchAdapter,
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Memory System Benchmark Tool"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--systems",
        type=str,
        default="mempalace",
        help="Comma-separated list of systems to test (mempalace,ogmemory,memsearch)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="locomo",
        help="Benchmark to run (locomo)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="../../../memory-eval/locomo/data/locomo10.json",
        help="Path to dataset",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve",
    )

    parser.add_argument(
        "--granularity",
        type=str,
        default="session",
        choices=["session", "dialog"],
        help="Corpus granularity",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="raw",
        help="Retrieval mode for MemPalace",
    )

    parser.add_argument(
        "--embed-model",
        type=str,
        default="default",
        help="Embedding model",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8090",
        help="oG-Memory API endpoint",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["json", "markdown", "both"],
        help="Output format",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        # 如果没有 yaml，使用默认配置
        return {}


def create_adapter(
    system_name: str,
    args
) -> Optional[object]:
    """创建适配器实例

    Args:
        system_name: 系统名称
        args: 命令行参数

    Returns:
        适配器实例或 None
    """
    if system_name == "mempalace":
        return MemPalaceAdapter(
            mode=args.mode,
            embed_model=args.embed_model,
            granularity=args.granularity,
        )
    elif system_name == "ogmemory":
        return OGMemoryAdapter(
            endpoint=args.endpoint,
        )
    elif system_name == "memsearch":
        return MemSearchAdapter(
            embed_provider="onnx",
        )
    else:
        print(f"Unknown system: {system_name}")
        return None


def run_benchmark(
    system_name: str,
    adapter: object,
    data_path: str,
    top_k: int,
    granularity: str,
) -> dict:
    """运行单个系统的基准测试

    Args:
        system_name: 系统名称
        adapter: 适配器实例
        data_path: 数据集路径
        top_k: 检索结果数量
        granularity: 语料粒度

    Returns:
        测试结果字典
    """
    print(f"\n{'='*60}")
    print(f"  Running {system_name} on LoCoMo")
    print(f"{'='*60}")

    # 初始化基准测试
    benchmark = LoCoMoBenchmark(
        data_path=data_path,
        adapter=adapter,
        granularity=granularity,
    )

    # 健康检查
    if not adapter.health_check():
        print(f"  [WARNING] Health check failed for {system_name}")

    # 加载数据
    print(f"  Loading data...")
    benchmark.load_data(data_path)

    # 准备语料并索引
    print(f"  Preparing corpus...")
    corpus = benchmark.prepare_corpus()
    print(f"  Indexing {len(corpus)} documents...")
    index_result = adapter.index(corpus)
    print(f"  Indexed in {index_result.duration_ms:.0f}ms")

    # 准备查询
    queries = benchmark.prepare_queries()
    print(f"  Running {len(queries)} queries...")

    # 性能监控
    monitor = PerformanceMonitor()
    monitor.metrics.index_time_ms = index_result.duration_ms

    # 逐个查询
    results = []
    for i, query in enumerate(queries):
        query_start = time.time()
        try:
            retrieved = adapter.search(query.question, top_k)
            query_time_ms = (time.time() - query_start) * 1000
            monitor.record_query_time(query_time_ms)

            eval_result = benchmark.evaluate_query(query, top_k, retrieved)
            eval_result.query_time_ms = query_time_ms
            results.append(eval_result)
        except Exception as e:
            query_time_ms = (time.time() - query_start) * 1000
            monitor.record_error(str(e))
            results.append({
                "query_id": query.id,
                "question": query.question,
                "error": str(e),
                "query_time_ms": query_time_ms,
            })

        if (i + 1) % 100 == 0:
            print(f"    Progress: {i + 1}/{len(queries)}")

    print(f"  Done!")

    # 计算聚合指标
    metrics = compute_aggregate_metrics(results, top_k)
    performance = monitor.get_metrics().to_dict()

    return {
        "config": adapter.get_config(),
        "metrics": metrics,
        "performance": performance,
        "results": results,
    }


def compute_aggregate_metrics(results: list, top_k: int) -> dict:
    """计算聚合指标"""
    if not results:
        return {}

    # 过滤有效结果
    valid_results = [r for r in results if hasattr(r, "retrieved_ids")]
    if not valid_results:
        return {}

    # 计算检索指标
    recall_values = {}
    ndcg_values = []
    mrr_values = []

    for k in [1, 5, 10]:
        recalls = []
        for r in valid_results:
            gt_set = set(r.ground_truth_ids)
            ret_ids = r.retrieved_ids[:k]
            recall = recall_at_k(ret_ids, gt_set, k)
            recalls.append(recall)
        recall_values[f"recall@{k}"] = sum(recalls) / len(recalls)

    for r in valid_results:
        gt_set = set(r.ground_truth_ids)
        ret_ids = r.retrieved_ids
        ndcg = ndcg_at_k(ret_ids, gt_set, top_k)
        ndcg_values.append(ndcg)

    recall_values[f"ndcg@{top_k}"] = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0

    # MRR
    retrieved_lists = [r.retrieved_ids for r in valid_results]
    gt_lists = [set(r.ground_truth_ids) for r in valid_results]
    recall_values["mrr"] = mrr(retrieved_lists, gt_lists)

    # 按类别分组
    by_category = {}
    category_groups: dict = {}
    for r in valid_results:
        cat = r.metadata.get("category", 0)
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(r)

    for cat, cat_results in category_groups.items():
        recalls = []
        for r in cat_results:
            gt_set = set(r.ground_truth_ids)
            recall = recall_at_k(r.retrieved_ids, gt_set, 10)
            recalls.append(recall)
        by_category[f"Cat_{cat}"] = {
            "count": len(cat_results),
            "recall@10": sum(recalls) / len(recalls) if recalls else 0,
        }

    return {
        **recall_values,
        "by_category": by_category,
    }


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"locomo_benchmark_{timestamp}"

    # 解析系统列表
    systems = [s.strip() for s in args.systems.split(",")]

    # 收集所有结果
    all_results = {
        "benchmark_name": "LoCoMo",
        "dataset_info": "LoCoMo (10 conversations, ~2000 QA pairs)",
        "timestamp": timestamp,
        "systems": {},
    }

    total_start = time.time()

    for system_name in systems:
        if system_name not in ADAPTER_REGISTRY:
            print(f"Skipping unknown system: {system_name}")
            continue

        # 创建适配器
        adapter = create_adapter(system_name, args)
        if adapter is None:
            continue

        # 运行基准测试
        system_results = run_benchmark(
            system_name=system_name,
            adapter=adapter,
            data_path=args.data,
            top_k=args.top_k,
            granularity=args.granularity,
        )

        all_results["systems"][system_name] = system_results

        # 重置适配器
        try:
            adapter.reset()
        except Exception:
            pass

    total_time = time.time() - total_start

    # 添加汇总信息
    all_results["summary"] = {
        "total_questions": sum(
            len(r.get("results", [])) for r in all_results["systems"].values()
        ),
        "systems_count": len(all_results["systems"]),
        "total_time_ms": total_time * 1000,
    }

    # 生成报告
    if args.format in ["json", "both"]:
        json_path = output_dir / f"{base_name}.json"
        generate_json_report(all_results, str(json_path))
        print(f"\nJSON report saved to: {json_path}")

    if args.format in ["markdown", "both"]:
        md_path = output_dir / f"{base_name}.md"
        generate_markdown_report(all_results, str(md_path))
        print(f"Markdown report saved to: {md_path}")

    print(f"\n{'='*60}")
    print(f"  Benchmark completed in {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()