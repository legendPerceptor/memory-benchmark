"""
Benchmark 基类
定义统一接口，让不同数据集可以被统一评测
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import json
import time

try:
    from .adapters.base import MemorySystemAdapter, RetrievalResult, IndexResult
except ImportError:
    from adapters.base import MemorySystemAdapter, RetrievalResult, IndexResult


@dataclass
class QueryEntry:
    """查询条目"""
    id: str
    question: str
    answer: str
    evidence_ids: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """单个查询的评估结果"""
    query_id: str
    question: str
    retrieved_ids: list[str]
    ground_truth_ids: list[str]
    retrieval_metrics: dict = field(default_factory=dict)
    qa_metrics: dict = field(default_factory=dict)
    query_time_ms: float = 0.0
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """完整基准测试结果"""
    benchmark_name: str
    system_name: str
    total_queries: int
    evaluation_results: list[EvaluationResult]
    retrieval_metrics: dict = field(default_factory=dict)
    qa_metrics: dict = field(default_factory=dict)
    performance_metrics: dict = field(default_factory=dict)
    index_time_ms: float = 0.0
    total_time_ms: float = 0.0


class Benchmark(ABC):
    """Benchmark 基类

    所有基准测试都需要实现以下接口:
    - load_data: 加载数据集
    - prepare_corpus: 准备要索引的语料
    - prepare_queries: 准备查询
    - evaluate_query: 评估单个查询
    - run: 运行完整基准测试
    """

    name: str = "base"

    def __init__(self, data_path: str, adapter: MemorySystemAdapter | None = None):
        """初始化基准测试

        Args:
            data_path: 数据集路径
            adapter: Memory 系统适配器
        """
        self.data_path = data_path
        self.adapter = adapter
        self.data: Any = None

    def load_data(self, path: str | None = None) -> Any:
        """加载数据集

        Args:
            path: 数据集路径，默认使用初始化时的路径

        Returns:
            加载的数据
        """
        path = path or self.data_path
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    @abstractmethod
    def prepare_corpus(self) -> list[dict]:
        """准备要索引的语料

        将数据集转换为适配器可以索引的格式:
        [{id, text, metadata}]

        Returns:
            语料列表
        """
        pass

    @abstractmethod
    def prepare_queries(self) -> list[QueryEntry]:
        """准备查询

        从数据集中提取查询:
        [{id, question, answer, evidence_ids, metadata}]

        Returns:
            查询列表
        """
        pass

    @abstractmethod
    def evaluate_query(
        self,
        query: QueryEntry,
        top_k: int,
        retrieved: list[RetrievalResult]
    ) -> EvaluationResult:
        """评估单个查询

        Args:
            query: 查询条目
            top_k: 检索结果数量
            retrieved: 检索结果

        Returns:
            评估结果
        """
        pass

    def run(
        self,
        adapter: MemorySystemAdapter | None = None,
        top_k: int = 10,
        progress_callback=None
    ) -> BenchmarkResult:
        """运行完整基准测试

        Args:
            adapter: Memory 系统适配器，如果未提供使用初始化时的适配器
            top_k: 检索结果数量
            progress_callback: 进度回调函数

        Returns:
            基准测试结果
        """
        adapter = adapter or self.adapter
        if adapter is None:
            raise ValueError("No adapter provided")

        start_time = time.time()

        # 1. 加载数据
        if self.data is None:
            self.load_data()

        # 2. 准备语料并索引
        corpus = self.prepare_corpus()
        index_start = time.time()
        index_result = adapter.index(corpus)
        index_time = (time.time() - index_start) * 1000

        # 3. 准备查询
        queries = self.prepare_queries()

        # 4. 逐个查询并评估
        results: list[EvaluationResult] = []
        for i, query in enumerate(queries):
            query_start = time.time()
            try:
                retrieved = adapter.search(query.question, top_k)
                query_time = (time.time() - query_start) * 1000
                eval_result = self.evaluate_query(query, top_k, retrieved)
                eval_result.query_time_ms = query_time
            except Exception as e:
                query_time = (time.time() - query_start) * 1000
                eval_result = EvaluationResult(
                    query_id=query.id,
                    question=query.question,
                    retrieved_ids=[],
                    ground_truth_ids=query.evidence_ids,
                    query_time_ms=query_time,
                    error=str(e)
                )
            results.append(eval_result)

            if progress_callback:
                progress_callback(i + 1, len(queries))

        total_time = (time.time() - start_time) * 1000

        return BenchmarkResult(
            benchmark_name=self.name,
            system_name=adapter.name,
            total_queries=len(queries),
            evaluation_results=results,
            index_time_ms=index_time,
            total_time_ms=total_time
        )
