"""
性能指标模块
提供延迟统计、内存采样等性能指标
"""

import time
import os
from typing import Optional
from dataclasses import dataclass, field
from statistics import mean, median, stdev

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    index_time_ms: float = 0.0
    query_times_ms: list[float] = field(default_factory=list)
    memory_samples_mb: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # 统计值
    avg_query_time_ms: float = 0.0
    median_query_time_ms: float = 0.0
    p95_query_time_ms: float = 0.0
    p99_query_time_ms: float = 0.0
    std_query_time_ms: float = 0.0

    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    def compute_stats(self) -> "PerformanceMetrics":
        """计算统计值"""
        if self.query_times_ms:
            sorted_times = sorted(self.query_times_ms)
            n = len(sorted_times)

            self.avg_query_time_ms = mean(self.query_times_ms)
            self.median_query_time_ms = median(self.query_times_ms)

            # 计算百分位数
            p95_idx = int(n * 0.95)
            p99_idx = int(n * 0.99)
            self.p95_query_time_ms = sorted_times[min(p95_idx, n - 1)]
            self.p99_query_time_ms = sorted_times[min(p99_idx, n - 1)]

            if n > 1:
                self.std_query_time_ms = stdev(self.query_times_ms)

        if self.memory_samples_mb:
            self.avg_memory_mb = mean(self.memory_samples_mb)
            self.peak_memory_mb = max(self.memory_samples_mb)

        self.total_queries = len(self.query_times_ms) + len(self.errors)
        self.successful_queries = len(self.query_times_ms)
        self.failed_queries = len(self.errors)

        return self

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "index_time_ms": self.index_time_ms,
            "avg_query_time_ms": self.avg_query_time_ms,
            "median_query_time_ms": self.median_query_time_ms,
            "p95_query_time_ms": self.p95_query_time_ms,
            "p99_query_time_ms": self.p99_query_time_ms,
            "std_query_time_ms": self.std_query_time_ms,
            "avg_memory_mb": self.avg_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
        }


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, process=None):
        """初始化性能监控器

        Args:
            process: 要监控的进程，默认为当前进程
        """
        if PSUTIL_AVAILABLE and process is None:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = process
        self.metrics = PerformanceMetrics()
        self._index_start_time: Optional[float] = None

    def start_index(self) -> None:
        """开始索引计时"""
        self._index_start_time = time.time()

    def end_index(self) -> float:
        """结束索引计时并记录

        Returns:
            索引耗时（毫秒）
        """
        if self._index_start_time is None:
            return 0.0

        duration_ms = (time.time() - self._index_start_time) * 1000
        self.metrics.index_time_ms = duration_ms
        self._index_start_time = None
        return duration_ms

    def record_query_time(self, duration_ms: float) -> None:
        """记录查询耗时

        Args:
            duration_ms: 查询耗时（毫秒）
        """
        self.metrics.query_times_ms.append(duration_ms)

    def sample_memory(self) -> float:
        """采样当前内存使用

        Returns:
            内存使用量（MB）
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)
            self.metrics.memory_samples_mb.append(mem_mb)
            return mem_mb
        except Exception:
            return 0.0

    def record_error(self, error: str) -> None:
        """记录错误

        Args:
            error: 错误信息
        """
        self.metrics.errors.append(error)

    def get_metrics(self) -> PerformanceMetrics:
        """获取性能指标

        Returns:
            性能指标数据类
        """
        return self.metrics.compute_stats()


class Timer:
    """计时器上下文管理器"""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    @property
    def elapsed_ms(self) -> float:
        """获取已过时间（毫秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return (end - self.start_time) * 1000


def compute_throughput(
    num_operations: int,
    total_time_ms: float
) -> dict:
    """计算吞吐量指标

    Args:
        num_operations: 操作数量
        total_time_ms: 总耗时（毫秒）

    Returns:
        吞吐量指标字典
    """
    if total_time_ms == 0:
        return {
            "ops_per_second": 0.0,
            "avg_ms_per_op": 0.0,
        }

    return {
        "ops_per_second": num_operations / (total_time_ms / 1000),
        "avg_ms_per_op": total_time_ms / num_operations,
    }