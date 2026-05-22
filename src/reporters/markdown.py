"""
Markdown 报告生成器
"""

from datetime import datetime
from typing import Any


def generate_markdown_report(results: dict, output_path: str) -> None:
    """生成 Markdown 格式的评测报告

    Args:
        results: 评测结果字典
        output_path: 输出文件路径
    """
    lines = []

    # 标题
    lines.extend([
        "# Memory System Benchmark Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Benchmark**: {results.get('benchmark_name', 'LoCoMo')}",
        f"**Dataset**: {results.get('dataset_info', 'LoCoMo (10 conversations)')}",
        "",
    ])

    # 汇总信息
    summary = results.get("summary", {})
    if summary:
        lines.extend([
            "## Summary",
            "",
            f"- **Total Questions**: {summary.get('total_questions', 'N/A')}",
            f"- **Systems Evaluated**: {summary.get('systems_count', 'N/A')}",
            f"- **Total Time**: {summary.get('total_time_ms', 0) / 1000:.1f}s",
            "",
        ])

    # 系统对比表
    systems = results.get("systems", {})
    if systems:
        lines.extend([
            "## Retrieval Metrics Comparison",
            "",
            "| System | R@1 | R@5 | R@10 | NDCG@10 | MRR |",
            "|--------|-----|-----|------|---------|-----|",
        ])

        for system_name, system_data in systems.items():
            metrics = system_data.get("metrics", {})
            config = system_data.get("config", {})
            mode = config.get("mode", "")

            system_label = f"{system_name}"
            if mode:
                system_label = f"{system_name} ({mode})"

            lines.append(
                f"| {system_label} | "
                f"{metrics.get('recall@1', 0):.3f} | "
                f"{metrics.get('recall@5', 0):.3f} | "
                f"{metrics.get('recall@10', 0):.3f} | "
                f"{metrics.get('ndcg@10', 0):.3f} | "
                f"{metrics.get('mrr', 0):.3f} |"
            )

        lines.append("")

    # 类别分解表
    category_data = results.get("category_breakdown", {})
    if category_data:
        lines.extend([
            "## Per-Category Breakdown (Recall@10)",
            "",
        ])

        # 获取所有类别
        all_categories = set()
        for system_data in systems.values():
            cat_metrics = system_data.get("metrics", {}).get("by_category", {})
            all_categories.update(cat_metrics.keys())

        if all_categories:
            # 表头
            lines.append("| Category | " + " | ".join(systems.keys()) + " |")
            lines.append("|----------|" + "|".join("---" for _ in systems) + "|")

            # 每行数据
            for category in sorted(all_categories):
                row = [f"| {category}"]
                for system_name, system_data in systems.items():
                    cat_metrics = system_data.get("metrics", {}).get("by_category", {})
                    recall = cat_metrics.get(category, {}).get("recall@10", 0)
                    row.append(f" {recall:.3f} |")
                lines.append("".join(row))

            lines.append("")

    # 性能指标表
    perf_data = results.get("performance", {})
    if perf_data:
        lines.extend([
            "## Performance Metrics",
            "",
            "| System | Index Time | Avg Query Time | P95 Query | Memory |",
            "|--------|-----------|----------------|-----------|--------|",
        ])

        for system_name, system_data in systems.items():
            perf = system_data.get("performance", {})
            perf_metrics = perf.get("metrics", {}) if isinstance(perf, dict) else perf

            lines.append(
                f"| {system_name} | "
                f"{perf_metrics.get('index_time_ms', 0) / 1000:.1f}s | "
                f"{perf_metrics.get('avg_query_time_ms', 0):.0f}ms | "
                f"{perf_metrics.get('p95_query_time_ms', 0):.0f}ms | "
                f"{perf_metrics.get('peak_memory_mb', 0):.0f}MB |"
            )

        lines.append("")

    # 详细配置
    lines.extend([
        "## System Configurations",
        "",
    ])

    for system_name, system_data in systems.items():
        config = system_data.get("config", {})
        lines.append(f"### {system_name}")
        lines.append("")
        lines.append("```yaml")
        for key, value in config.items():
            lines.append(f"  {key}: {value}")
        lines.append("```")
        lines.append("")

    # 保存到文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_comparison_table(results_list: list[dict]) -> str:
    """生成对比表格

    Args:
        results_list: 结果字典列表

    Returns:
        Markdown 格式的表格
    """
    lines = []

    # 获取所有系统和指标
    all_systems = set()
    all_metrics = set()
    for results in results_list:
        for system in results.get("systems", {}):
            all_systems.add(system)
            metrics = results.get("systems", {}).get(system, {}).get("metrics", {})
            all_metrics.update(metrics.keys())

    # 排序
    sorted_systems = sorted(all_systems)
    sorted_metrics = sorted(all_metrics)

    # 表头
    lines.append("| Metric | " + " | ".join(sorted_systems) + " |")
    lines.append("|--------|" + "|".join("---" for _ in sorted_systems) + "|")

    # 每行数据
    for metric in sorted_metrics:
        row = [f"| {metric}"]
        for system in sorted_systems:
            value = "N/A"
            for results in results_list:
                system_data = results.get("systems", {}).get(system, {})
                metrics = system_data.get("metrics", {})
                if metric in metrics:
                    val = metrics[metric]
                    if isinstance(val, float):
                        value = f"{val:.3f}"
                    else:
                        value = str(val)
                    break
            row.append(f" {value} |")
        lines.append("".join(row))

    return "\n".join(lines)