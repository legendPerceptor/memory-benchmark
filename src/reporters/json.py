"""
JSON 报告生成器
"""

import json
from datetime import datetime
from typing import Any


def generate_json_report(results: dict, output_path: str) -> None:
    """生成 JSON 格式的评测报告

    Args:
        results: 评测结果字典
        output_path: 输出文件路径
    """
    # 构建报告结构
    report = {
        "report_type": "memory_benchmark",
        "generated_at": datetime.now().isoformat(),
        "benchmark": results.get("benchmark_name", "unknown"),
        "summary": results.get("summary", {}),
        "systems": {},
    }

    # 添加各系统的详细结果
    for system_name, system_results in results.get("systems", {}).items():
        report["systems"][system_name] = {
            "config": system_results.get("config", {}),
            "metrics": system_results.get("metrics", {}),
            "performance": system_results.get("performance", {}),
        }

    # 保存到文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def load_json_report(input_path: str) -> dict:
    """加载 JSON 报告

    Args:
        input_path: 报告文件路径

    Returns:
        报告内容字典
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_reports(report_paths: list[str]) -> dict:
    """对比多个报告

    Args:
        report_paths: 报告文件路径列表

    Returns:
        对比结果字典
    """
    reports = [load_json_report(path) for path in report_paths]

    comparison = {
        "report_count": len(reports),
        "benchmarks": list(set(r.get("benchmark", "unknown") for r in reports)),
        "systems": {},
    }

    # 收集所有系统
    all_systems = set()
    for report in reports:
        all_systems.update(report.get("systems", {}).keys())

    # 对比每个系统
    for system in all_systems:
        system_metrics = {}
        for report in reports:
            system_data = report.get("systems", {}).get(system, {})
            metrics = system_data.get("metrics", {})
            for metric_name, value in metrics.items():
                if metric_name not in system_metrics:
                    system_metrics[metric_name] = []
                system_metrics[metric_name].append(value)

        comparison["systems"][system] = {
            "reports_count": len([r for r in reports if system in r.get("systems", {})]),
            "metrics": {
                name: {
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "values": values,
                }
                for name, values in system_metrics.items()
            },
        }

    return comparison