"""
oG-Memory 适配器
通过 HTTP API 调用 oG-Memory 服务
"""

import json
import time
import urllib.request
import urllib.error
from typing import Optional

try:
    from ..base import MemorySystemAdapter, RetrievalResult, IndexResult
except ImportError:
    from adapters.base import MemorySystemAdapter, RetrievalResult, IndexResult


class OGMemoryAdapter(MemorySystemAdapter):
    """oG-Memory Memory System Adapter

    通过 HTTP API 与 oG-Memory 服务通信
    """

    name: str = "ogmemory"

    def __init__(
        self,
        endpoint: str = "http://localhost:8090",
        api_key: str = "default",
        confidence_threshold: float = 0.5,
        timeout: int = 30,
    ):
        """初始化 oG-Memory 适配器

        Args:
            endpoint: oG-Memory API 端点
            api_key: API 密钥
            confidence_threshold: 记忆写入置信度阈值
            timeout: 请求超时（秒）
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout

        self._session_id: Optional[str] = None
        self._user_id: Optional[str] = None
        self._account_id: Optional[str] = None

    def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[dict] = None,
    ) -> dict:
        """发送 HTTP 请求

        Args:
            method: HTTP 方法
            path: API 路径
            data: 请求数据

        Returns:
            响应数据

        Raises:
            Exception: 请求失败时抛出异常
        """
        url = f"{self.endpoint}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

        body = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise Exception(f"HTTP {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Connection error: {e.reason}")

    def health_check(self) -> bool:
        """检查 oG-Memory 服务是否可用"""
        try:
            self._make_request("GET", "/api/v1/health")
            return True
        except Exception:
            return False

    def _init_session(self) -> None:
        """初始化会话"""
        if self._session_id is None:
            self._session_id = f"bench_{int(time.time() * 1000)}"
            self._user_id = f"user_bench"
            self._account_id = f"account_bench"

    def index(self, corpus: list[dict]) -> IndexResult:
        """索引语料

        将对话会话写入 oG-Memory

        Args:
            corpus: 语料列表 [{id, text, metadata}]

        Returns:
            IndexResult: 索引结果
        """
        import time
        start_time = time.time()

        self._init_session()

        total_indexed = 0
        for doc in corpus:
            # 准备消息
            messages = [{"role": "user", "content": doc["text"]}]

            try:
                self._make_request(
                    "POST",
                    "/api/v1/after_turn",
                    {
                        "messages": messages,
                        "sessionId": self._session_id,
                        "userId": self._user_id,
                        "accountId": self._account_id,
                        "confidenceThreshold": self.confidence_threshold,
                    },
                )
                total_indexed += 1
            except Exception as e:
                # 记录错误但继续
                pass

        duration_ms = (time.time() - start_time) * 1000

        return IndexResult(
            indexed_count=total_indexed,
            duration_ms=duration_ms,
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """检索相关文档

        使用 oG-Memory compose API 进行检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            list[RetrievalResult]: 检索结果列表
        """
        self._init_session()

        # 使用 compose API
        messages = [{"role": "user", "content": query}]

        try:
            result = self._make_request(
                "POST",
                "/api/v1/compose",
                {
                    "messages": messages,
                    "sessionId": self._session_id,
                    "userId": self._user_id,
                    "accountId": self._account_id,
                    "topK": top_k,
                },
            )
        except Exception as e:
            raise Exception(f"Compose failed: {e}")

        # 解析检索结果
        retrieval_results = []

        # 从 retrievedEvidence 中提取结果
        evidence = result.get("retrievedEvidence", "")
        if evidence:
            # 简单处理：将 evidence 作为单一结果返回
            retrieval_results.append(
                RetrievalResult(
                    id=f"evidence_0",
                    text=evidence,
                    score=1.0,
                    metadata={},
                )
            )

        # 从 stats 中获取 hit_count
        stats = result.get("stats", {})
        hit_count = stats.get("hit_count", 0)

        # 如果有会话上下文，也作为结果返回
        episodic = result.get("episodicContext", "")
        if episodic:
            retrieval_results.append(
                RetrievalResult(
                    id=f"episodic_0",
                    text=episodic,
                    score=0.9,
                    metadata={"type": "episodic"},
                )
            )

        identity = result.get("identityContext", "")
        if identity:
            retrieval_results.append(
                RetrievalResult(
                    id=f"identity_0",
                    text=identity,
                    score=0.8,
                    metadata={"type": "identity"},
                )
            )

        return retrieval_results[:top_k]

    def reset(self) -> None:
        """重置会话"""
        if self._session_id:
            try:
                self._make_request("POST", f"/api/v1/dispose", {
                    "sessionId": self._session_id,
                    "userId": self._user_id,
                    "accountId": self._account_id,
                })
            except Exception:
                pass

        self._session_id = None

    def get_config(self) -> dict:
        """获取适配器配置"""
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "confidence_threshold": self.confidence_threshold,
        }