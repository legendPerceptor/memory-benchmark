"""
MemPalace 适配器
使用 MemPalace 进行对话记忆检索
"""

import sys
import re
import string
import tempfile
from pathlib import Path
from collections import Counter
from typing import Optional

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# 添加 mempalace 路径
mempalace_path = Path(__file__).parent.parent.parent.parent / "mempalace"
sys.path.insert(0, str(mempalace_path))

try:
    from .base import MemorySystemAdapter, RetrievalResult, IndexResult
except ImportError:
    from adapters.base import MemorySystemAdapter, RetrievalResult, IndexResult


# ── 嵌入模型 ────────────────────────────────────────────────────────────────

_fastembed_model = None
_api_embedder = None


def _get_embedder(model_name: str):
    """Lazy-load a fastembed model. Cached globally after first load."""
    global _fastembed_model
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding
            _fastembed_model = TextEmbedding(model_name=model_name)
        except ImportError:
            raise ImportError("fastembed not installed — pip install fastembed")
    return _fastembed_model


def _get_api_embedder(provider: str, model: str, api_key: str, base_url: str = None):
    """Get API-based embedder (OpenAI-compatible or Volcengine)."""
    global _api_embedder

    # Check if we can reuse cached embedder with same config
    cache_key = (provider, model, api_key, base_url)
    if _api_embedder and getattr(_api_embedder, '_cache_key', None) == cache_key:
        return _api_embedder

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai not installed — pip install openai")

    # Build base_url for API call
    if provider == "volcengine":
        if base_url:
            emb_base_url = base_url
        else:
            emb_base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
    elif base_url:
        emb_base_url = base_url
    else:
        emb_base_url = "https://api.openai.com/v1"

    client_kwargs = {"api_key": api_key, "base_url": emb_base_url}
    sync_client = AsyncOpenAI(**client_kwargs)

    class APIEmbedder:
        """Wrapper for API-based embedding."""
        _cache_key = cache_key

        def __init__(self, model_name: str, dimension: int = 1024):
            self.model = model_name
            self.dimension = dimension

        def embed(self, texts: list) -> list:
            """Embed texts synchronously."""
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            future = self._embed_async(texts)
            return loop.run_until_complete(future)

        async def _embed_async(self, texts: list) -> list:
            resp = await sync_client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format="float"
            )
            return [item.embedding for item in resp.data]

    # Detect dimension for common models
    dimension_map = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    dimension = dimension_map.get(model, 1024)

    _api_embedder = APIEmbedder(model, dimension)
    return _api_embedder


def _embed(texts: list, embed_model: str, embed_provider: str = None, api_key: str = None, embed_base_url: str = None) -> Optional[list]:
    """Embed a list of texts.

    Args:
        texts: List of texts to embed
        embed_model: Model name (used for API or local)
        embed_provider: Provider type ("openai", "volcengine", or None for local)
        api_key: API key for remote embedding
        embed_base_url: Custom API base URL
    """
    # Use API embedding if provider is specified
    if embed_provider and embed_provider != "onnx":
        if not api_key:
            raise ValueError(f"API key required for embed_provider={embed_provider}")
        embedder = _get_api_embedder(embed_provider, embed_model, api_key, embed_base_url)
        return [vec for vec in embedder.embed(texts)]

    # Use local fastembed (default behavior)
    if not embed_model or embed_model == "default":
        return None
    embedder = _get_embedder(embed_model)
    return [vec.tolist() for vec in embedder.embed(texts)]


# ── 文本标准化 ─────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Normalize answer for F1 comparison."""
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return s.lower().strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with normalization."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


# ── MemPalace 适配器 ───────────────────────────────────────────────────────

class MemPalaceAdapter(MemorySystemAdapter):
    """MemPalace Memory System Adapter

    使用 ChromaDB 进行向量存储，支持多种检索模式
    """

    name: str = "mempalace"

    def __init__(
        self,
        mode: str = "raw",
        embed_model: str = "default",
        embed_provider: str = None,
        embed_base_url: str = None,
        api_key: str = None,
        collection_name: str = "locomo_test",
        granularity: str = "session",
    ):
        """初始化 MemPalace 适配器

        Args:
            mode: 检索模式 (raw/hybrid/aaak/rooms/palace)
            embed_model: 嵌入模型名称
            embed_provider: 嵌入提供商 (openai, volcengine, onnx)
            embed_base_url: 自定义 API base URL
            api_key: API 密钥
            collection_name: ChromaDB collection 名称
            granularity: 语料粒度 (session/dialog)
        """
        self.mode = mode
        self.embed_model = embed_model
        self.embed_provider = embed_provider
        self.embed_base_url = embed_base_url
        self.api_key = api_key
        self.collection_name = collection_name
        self.granularity = granularity

        self._client = None
        self._collection = None
        self._temp_dir: Optional[str] = None

    def health_check(self) -> bool:
        """检查 ChromaDB 是否可用"""
        try:
            if self._client is None:
                self._init_client()
            return True
        except Exception:
            return False

    def _init_client(self) -> None:
        """初始化 ChromaDB 客户端"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")
        self._temp_dir = tempfile.mkdtemp()
        self._client = chromadb.PersistentClient(path=self._temp_dir)
        self._collection = self._client.create_collection(
            self.collection_name,
            metadata={"hnsn:space_complexity": 2}
        )

    def index(self, corpus: list[dict]) -> IndexResult:
        """索引语料

        Args:
            corpus: 语料列表 [{id, text, metadata}]

        Returns:
            IndexResult: 索引结果
        """
        import time
        start_time = time.time()

        if self._client is None:
            self._init_client()

        # 准备数据
        ids = []
        documents = []
        metadatas = []

        for doc in corpus:
            ids.append(doc["id"])
            documents.append(doc["text"])
            metadatas.append(doc.get("metadata", {}))

        # 批量添加
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        # 可选：预计算嵌入（API 或本地）
        if self.embed_model != "default" or self.embed_provider:
            embeddings = _embed(
                documents,
                self.embed_model,
                self.embed_provider,
                self.api_key,
                self.embed_base_url,
            )
            if embeddings:
                # 重新添加（使用嵌入）
                self._collection.delete()
                self._collection = self._client.create_collection(
                    self.collection_name,
                    metadata={"hnsn:space_complexity": 2}
                )
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )

        duration_ms = (time.time() - start_time) * 1000

        return IndexResult(
            indexed_count=len(corpus),
            duration_ms=duration_ms,
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """检索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            list[RetrievalResult]: 检索结果列表
        """
        if self._collection is None:
            raise ValueError("Collection not initialized. Call index() first.")

        # 准备查询
        q_emb = _embed(
            [query],
            self.embed_model,
            self.embed_provider,
            self.api_key,
            self.embed_base_url,
        )
        kwargs = dict(
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )
        if q_emb is not None:
            kwargs["query_embeddings"] = q_emb
        else:
            kwargs["query_texts"] = [query]

        # 执行查询
        results = self._collection.query(**kwargs)

        # 转换结果
        retrieval_results = []
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}

            retrieval_results.append(
                RetrievalResult(
                    id=doc_id,
                    text=document,
                    score=1.0 / (1.0 + distance),  # 转换为相似度分数
                    metadata=metadata,
                )
            )

        # Hybrid 模式重排序
        if self.mode == "hybrid":
            retrieval_results = self._hybrid_rerank(query, retrieval_results)

        return retrieval_results

    def _hybrid_rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Hybrid 模式重排序

        基于关键词重叠对向量检索结果进行重排序
        """
        # 提取查询关键词
        query_words = self._extract_keywords(query)

        # 计算融合分数
        for r in results:
            kw_overlap = self._keyword_overlap(query_words, r.text)
            fused_score = r.score * (1.0 - 0.5 * kw_overlap)
            r.score = fused_score

        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _extract_keywords(self, text: str) -> set:
        """提取关键词"""
        stop_words = {
            "what", "when", "where", "who", "how", "which",
            "did", "do", "was", "were", "have", "has", "had",
            "is", "are", "the", "a", "an", "my", "me", "i",
            "you", "your", "their", "it", "its", "in", "on",
            "at", "to", "for", "of", "with", "by", "from",
        }
        words = re.findall(r"\w+", text.lower())
        return {w for w in words if w not in stop_words and len(w) > 2}

    def _keyword_overlap(self, keywords: set, text: str) -> float:
        """计算关键词重叠率"""
        text_lower = text.lower()
        hits = sum(1 for kw in keywords if kw in text_lower)
        return hits / len(keywords) if keywords else 0.0

    def reset(self) -> None:
        """清空索引"""
        if self._client is not None and self._collection is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass

        if self._temp_dir:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        self._client = None
        self._collection = None

    def get_config(self) -> dict:
        """获取适配器配置"""
        return {
            "name": self.name,
            "mode": self.mode,
            "embed_model": self.embed_model,
            "embed_provider": self.embed_provider,
            "embed_base_url": self.embed_base_url,
            "collection_name": self.collection_name,
            "granularity": self.granularity,
        }