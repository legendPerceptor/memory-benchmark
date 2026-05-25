"""
memsearch 适配器
使用 memsearch 进行语义搜索
"""

import sys
import time
import tempfile
import asyncio
from pathlib import Path
from typing import Optional

# 添加 memsearch 路径
memsearch_path = Path(__file__).parent.parent.parent.parent / "memsearch"
sys.path.insert(0, str(memsearch_path))

try:
    from .base import MemorySystemAdapter, RetrievalResult, IndexResult
except ImportError:
    from adapters.base import MemorySystemAdapter, RetrievalResult, IndexResult


class MemSearchAdapter(MemorySystemAdapter):
    """memsearch Memory System Adapter

    使用 MemSearch 进行语义搜索
    """

    name: str = "memsearch"

    def __init__(
        self,
        embed_provider: str = "onnx",
        embed_model: str = "BAAI/bge-m3",
        embed_base_url: str = None,
        api_key: str = None,
        collection: str = "locomo_test",
        top_k: int = 10,
    ):
        """初始化 memsearch 适配器

        Args:
            embed_provider: 嵌入提供者 (onnx/openai/google/etc.)
            embed_model: 嵌入模型名称
            embed_base_url: 自定义 API base URL
            api_key: API 密钥
            collection: Milvus collection 名称
            top_k: 默认返回结果数量
        """
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.embed_base_url = embed_base_url
        self.api_key = api_key
        self.collection = collection
        self.default_top_k = top_k

        self._memsearch = None
        self._temp_dir: Optional[str] = None
        self._indexed_paths: list[str] = []

    def health_check(self) -> bool:
        """检查 memsearch 是否可用"""
        try:
            from memsearch import MemSearch
            return True
        except ImportError:
            return False

    def _init_memsearch(self) -> None:
        """初始化 MemSearch 实例"""
        if self._memsearch is not None:
            return

        from memsearch import MemSearch

        # 创建临时目录存储 markdown 文件
        self._temp_dir = tempfile.mkdtemp()

        self._memsearch = MemSearch(
            paths=[self._temp_dir],
            embedding_provider=self.embed_provider,
            embedding_model=self.embed_model,
            embedding_base_url=self.embed_base_url,
            embedding_api_key=self.api_key,
            collection=self.collection,
        )

    def index(self, corpus: list[dict]) -> IndexResult:
        """索引语料

        将语料写入临时 markdown 文件，然后使用 memsearch 索引

        Args:
            corpus: 语料列表 [{id, text, metadata}]

        Returns:
            IndexResult: 索引结果
        """
        import time
        start_time = time.time()

        self._init_memsearch()

        # 将语料写入 markdown 文件
        for doc in corpus:
            doc_id = doc["id"]
            text = doc["text"]
            metadata = doc.get("metadata", {})

            # 创建 markdown 文件
            date = metadata.get("date", "")
            filename = f"{doc_id}.md"
            filepath = Path(self._temp_dir) / filename

            # 格式化为 markdown
            content = f"# {doc_id}\n\n"
            if date:
                content += f"**Date**: {date}\n\n"
            content += text

            filepath.write_text(content, encoding="utf-8")
            self._indexed_paths.append(str(filepath))

        # 异步索引
        if self._memsearch:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(self._memsearch.index())
            except Exception:
                # 索引可能失败，但继续
                pass

        duration_ms = (time.time() - start_time) * 1000

        return IndexResult(
            indexed_count=len(corpus),
            duration_ms=duration_ms,
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """检索相关文档

        使用 memsearch 进行语义搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            list[RetrievalResult]: 检索结果列表
        """
        if self._memsearch is None:
            self._init_memsearch()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(
                self._memsearch.search(query, top_k=top_k)
            )
        except Exception as e:
            raise Exception(f"MemSearch search failed: {e}")

        # 转换结果
        retrieval_results = []
        for i, result in enumerate(results):
            retrieval_results.append(
                RetrievalResult(
                    id=result.get("chunk_hash", f"result_{i}"),
                    text=result.get("content", ""),
                    score=result.get("score", 0.0),
                    metadata={
                        "source": result.get("source", ""),
                        "start_line": result.get("start_line", 0),
                        "end_line": result.get("end_line", 0),
                    },
                )
            )

        return retrieval_results

    def reset(self) -> None:
        """重置/清空索引"""
        if self._memsearch is not None and self._temp_dir:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)

        self._memsearch = None
        self._temp_dir = None
        self._indexed_paths = []

    def get_config(self) -> dict:
        """获取适配器配置"""
        return {
            "name": self.name,
            "embed_provider": self.embed_provider,
            "embed_model": self.embed_model,
            "embed_base_url": self.embed_base_url,
            "collection": self.collection,
        }