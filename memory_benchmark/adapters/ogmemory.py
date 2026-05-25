"""
oG-Memory 适配器
通过 HTTP API 调用 oG-Memory 服务
"""

import asyncio
import json
import time
import urllib.request
import urllib.error
from typing import Optional

from .base import MemorySystemAdapter, RetrievalResult, IndexResult


class OGMemoryAdapter(MemorySystemAdapter):
    """oG-Memory Memory System Adapter

    通过 HTTP API 与 oG-Memory 服务通信
    支持检索模式和生成模式
    """

    name: str = "ogmemory"

    def __init__(
        self,
        endpoint: str = "http://localhost:8090",
        api_key: str = "default",
        confidence_threshold: float = 0.5,
        timeout: int = 30,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str = None,
        llm_base_url: str = None,
        generation_mode: bool = False,
    ):
        """初始化 oG-Memory 适配器

        Args:
            endpoint: oG-Memory API 端点
            api_key: API 密钥
            confidence_threshold: 记忆写入置信度阈值
            timeout: 请求超时（秒）
            llm_model: LLM 模型名称
            llm_api_key: LLM API 密钥
            llm_base_url: 自定义 LLM API URL
            generation_mode: 是否启用生成模式（检索+LLM生成答案）
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.generation_mode = generation_mode

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

        将对话会话按时间顺序逐轮写入 oG-Memory。
        每个 corpus 项包含完整的 session，对话按 dialog 顺序逐轮写入，
        让 oG-Memory 自己决定提取哪些记忆。

        注意：oG-Memory 的 after_turn API 需要 LLM 来提取记忆，
        LLM 配置在 oG-Memory 服务端（config/ogmem.yaml 或环境变量）。

        Args:
            corpus: 语料列表 [{id, text, metadata}]
                   text 格式: JSON 字符串，包含 session 对话
                   metadata 可能包含: sample_id, session_num 等

        Returns:
            IndexResult: 索引结果
        """
        import time
        start_time = time.time()

        self._init_session()

        total_indexed = 0
        total_dialogs = 0

        for doc in corpus:
            doc_id = doc["id"]
            text = doc["text"]
            metadata = doc.get("metadata", {})

            try:
                # 解析对话内容
                # text 可能是 JSON 格式的对话列表，或原始文本
                dialogs = self._parse_dialogs(text)

                # 按时间顺序逐轮写入
                for dialog in dialogs:
                    speaker = dialog.get("speaker", "user")
                    content = dialog.get("text", "")

                    # 转换为消息格式
                    # 假设对话是 user 和 assistant 交替
                    role = "user" if speaker != "assistant" else "assistant"

                    messages = [{"role": role, "content": content}]

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
                        total_dialogs += 1
                    except Exception:
                        # 单轮失败继续
                        pass

                total_indexed += 1
            except Exception as e:
                # 解析失败，记录错误但继续处理其他文档
                pass

        duration_ms = (time.time() - start_time) * 1000

        return IndexResult(
            indexed_count=total_indexed,
            duration_ms=duration_ms,
        )

    def _parse_dialogs(self, text: str) -> list[dict]:
        """解析对话文本为 dialog 列表

        支持两种格式：
        1. JSON 格式: [{"speaker": "...", "text": "...", "dia_id": "..."}]
        2. 原始文本格式: 直接作为单轮对话处理

        Args:
            text: 对话文本

        Returns:
            dialog 列表
        """
        import json

        # 尝试 JSON 解析
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # 可能包含 sessions 或其他结构
                return self._parse_lohemo_structure(data)
        except (json.JSONDecodeError, TypeError):
            pass

        # 作为原始文本处理
        return [{"speaker": "user", "text": text, "dia_id": "D0:1"}]

    def _parse_lohemo_structure(self, data: dict) -> list[dict]:
        """解析 LoCoMo 数据结构

        LoCoMo 数据格式:
        {
            "session_1": [{"speaker": "...", "text": "...", "dia_id": "D1:1"}, ...],
            "session_2": [...],
            ...
        }

        Returns:
            按时间顺序排列的对话列表
        """
        dialogs = []
        session_nums = []

        # 找出所有 session
        for key in data.keys():
            if key.startswith("session_") and not key.endswith("_date_time"):
                try:
                    num = int(key.split("_")[1])
                    session_nums.append(num)
                except (ValueError, IndexError):
                    pass

        # 按 session 顺序处理
        for session_num in sorted(session_nums):
            session_key = f"session_{session_num}"
            session_dialogs = data.get(session_key, [])
            for dialog in session_dialogs:
                dialogs.append(dialog)

        return dialogs

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

    def _generate_answer(self, query: str, context: str) -> str:
        """使用 LLM 根据检索上下文生成答案

        Args:
            query: 问题
            context: 检索到的上下文

        Returns:
            生成的答案
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai not installed — pip install openai")

        # 构建 API 配置
        api_key = self.llm_api_key or self.api_key
        base_url = self.llm_base_url or "https://api.openai.com/v1"

        sync_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer the question based on the context above. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question." """

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                sync_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500,
                )
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"LLM generation failed: {e}")

    def query_with_generation(self, query: str, top_k: int = 10) -> dict:
        """检索上下文并生成答案

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            dict: 包含检索结果和生成答案
        """
        self._init_session()

        # 检索上下文
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

        # 提取检索上下文
        evidence = result.get("retrievedEvidence", "")
        episodic = result.get("episodicContext", "")
        identity = result.get("identityContext", "")
        session = result.get("sessionContext", "")

        # 合并所有上下文
        context_parts = []
        if identity:
            context_parts.append(f"Identity Information:\n{identity}")
        if episodic:
            context_parts.append(f"Past History:\n{episodic}")
        if session:
            context_parts.append(f"Current Session:\n{session}")
        if evidence:
            context_parts.append(f"Retrieved Information:\n{evidence}")

        full_context = "\n\n".join(context_parts)

        # 构建检索结果（用于评估）
        retrieval_results = []
        if evidence:
            retrieval_results.append(
                RetrievalResult(
                    id="evidence_0",
                    text=evidence,
                    score=1.0,
                    metadata={"type": "evidence"},
                )
            )
        if episodic:
            retrieval_results.append(
                RetrievalResult(
                    id="episodic_0",
                    text=episodic,
                    score=0.9,
                    metadata={"type": "episodic"},
                )
            )
        if identity:
            retrieval_results.append(
                RetrievalResult(
                    id="identity_0",
                    text=identity,
                    score=0.8,
                    metadata={"type": "identity"},
                )
            )

        # 使用 LLM 生成答案
        generated_answer = self._generate_answer(query, full_context)

        return {
            "retrieval_results": retrieval_results[:top_k],
            "generated_answer": generated_answer,
            "context": full_context,
        }

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
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generation_mode": self.generation_mode,
        }