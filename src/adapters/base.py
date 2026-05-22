"""
Memory System Adapter 基类
定义统一接口，让不同 memory 系统可以被统一评测
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """检索结果"""
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexResult:
    """索引结果"""
    indexed_count: int
    duration_ms: float
    error: str | None = None


@dataclass
class QueryResult:
    """查询结果"""
    query: str
    retrieved: list[RetrievalResult]
    duration_ms: float
    error: str | None = None


class MemorySystemAdapter(ABC):
    """Memory System Adapter 基类

    所有 memory 系统适配器都需要实现以下接口:
    - health_check: 检查服务是否可用
    - index: 索引语料
    - search: 检索相关文档
    - reset: 清空索引
    """

    name: str = "base"

    @abstractmethod
    def health_check(self) -> bool:
        """检查服务是否可用

        Returns:
            bool: 服务是否正常运行
        """
        pass

    @abstractmethod
    def index(self, corpus: list[dict]) -> IndexResult:
        """索引语料

        Args:
            corpus: 语料列表，每个元素包含:
                - id: 文档ID (str)
                - text: 文档内容 (str)
                - metadata: 元数据 (dict, optional)

        Returns:
            IndexResult: 索引结果，包含索引数量和耗时
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """检索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            list[RetrievalResult]: 检索结果列表，按相关性排序
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """清空索引，重置系统状态"""
        pass

    def get_config(self) -> dict:
        """获取适配器配置信息

        Returns:
            dict: 配置信息字典
        """
        return {"name": self.name}
