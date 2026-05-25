"""
LoCoMo 基准测试模块
评估 memory 系统在 LoCoMo 对话长期记忆数据集上的表现
"""

import json
from typing import Optional
from collections import Counter

from .base import Benchmark, QueryEntry, EvaluationResult
try:
    from .adapters.base import MemorySystemAdapter, RetrievalResult
except ImportError:
    from adapters.base import MemorySystemAdapter, RetrievalResult
try:
    from .metrics.retrieval import recall_at_k, ndcg_at_k, mrr
except ImportError:
    from metrics.retrieval import recall_at_k, ndcg_at_k, mrr
try:
    from .metrics.qa import f1_score, exact_match
except ImportError:
    from metrics.qa import f1_score, exact_match


CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Temporal-inference",
    4: "Open-domain",
    5: "Adversarial",
}


def load_conversation_sessions(conversation: dict, session_summaries: Optional[dict] = None):
    """从 LoCoMo 对话字典中提取会话

    Args:
        conversation: 对话字典
        session_summaries: 可选的会话摘要字典

    Returns:
        会话列表
    """
    sessions = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if key not in conversation:
            break
        dialogs = conversation[key]
        date = conversation.get(date_key, "")
        summary = ""
        if session_summaries:
            summary = session_summaries.get(f"session_{session_num}_summary", "")
        sessions.append({
            "session_num": session_num,
            "date": date,
            "dialogs": dialogs,
            "summary": summary,
        })
        session_num += 1
    return sessions


def build_corpus_from_sessions(
    sessions: list[dict],
    granularity: str = "session"
) -> tuple[list[str], list[str], list[str]]:
    """从会话构建检索语料

    Args:
        sessions: 会话列表
        granularity: 粒度 (session/dialog)

    Returns:
        (corpus_texts, corpus_ids, corpus_timestamps)
    """
    corpus = []
    corpus_ids = []
    corpus_timestamps = []

    for sess in sessions:
        if granularity == "session":
            # 合并所有对话为一个文档
            texts = []
            for d in sess["dialogs"]:
                speaker = d.get("speaker", "?")
                text = d.get("text", "")
                texts.append(f"{speaker} said, \"{text}\"")
            doc = "\n".join(texts)
            corpus.append(doc)
            corpus_ids.append(f"session_{sess['session_num']}")
            corpus_timestamps.append(sess["date"])
        else:
            # 每个对话轮次一个文档
            for d in sess["dialogs"]:
                dia_id = d.get("dia_id", f"D{sess['session_num']}:?")
                speaker = d.get("speaker", "?")
                text = d.get("text", "")
                doc = f'{speaker} said, "{text}"'
                corpus.append(doc)
                corpus_ids.append(dia_id)
                corpus_timestamps.append(sess["date"])

    return corpus, corpus_ids, corpus_timestamps


def resolve_evidence_ids(evidence_refs: list, sample: dict) -> list[str]:
    """解析证据引用为实际的文档 ID

    Args:
        evidence_refs: 证据引用列表 (如 ["D1:3", "D2:1"])
        sample: LoCoMo 样本数据

    Returns:
        实际的文档 ID 列表
    """
    resolved = []

    for ref in evidence_refs:
        # 格式: D1:3 表示 session_1 的第 3 个对话
        if ":" in ref:
            try:
                parts = ref.split(":")
                session_num = int(parts[0][1:])  # D1 -> 1
                dialog_num = int(parts[1])

                session_key = f"session_{session_num}"
                if session_key in sample["conversation"]:
                    dialogs = sample["conversation"][session_key]
                    if dialog_num <= len(dialogs):
                        resolved.append(dialogs[dialog_num - 1].get("dia_id", ref))
                    else:
                        # 如果索引超出，使用 session ID
                        resolved.append(f"session_{session_num}")
                else:
                    resolved.append(f"session_{session_num}")
            except (ValueError, IndexError):
                # 解析失败，使用引用作为 ID
                resolved.append(ref)
        else:
            resolved.append(ref)

    return resolved


class LoCoMoBenchmark(Benchmark):
    """LoCoMo 对话长期记忆基准测试

    数据集: 10 个对话，每个对话包含多个会话，约 200 个 QA 对
    评估维度: 检索召回、问答准确率
    """

    name: str = "locomo"

    def __init__(
        self,
        data_path: str,
        adapter: Optional[MemorySystemAdapter] = None,
        granularity: str = "session",
    ):
        """初始化 LoCoMo 基准测试

        Args:
            data_path: locomo10.json 数据集路径
            adapter: Memory 系统适配器
            granularity: 语料粒度 (session/dialog)
        """
        super().__init__(data_path, adapter)
        self.granularity = granularity
        self._corpus_ids: list[str] = []
        self._corpus_timestamps: list[str] = []

    def load_data(self, path: str) -> list[dict]:
        """加载 LoCoMo 数据集"""
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def prepare_corpus(self) -> list[dict]:
        """准备要索引的语料

        将 LoCoMo 对话转换为文档列表。
        注意：对于 oG-Memory，应该输出原始对话结构，让它按时间顺序逐轮写入。

        Returns:
            语料列表 [{id, text, metadata}]
                   - id: 样本 ID
                   - text: JSON 格式的对话结构
                   - metadata: 样本元信息
        """
        import json

        corpus = []

        for sample in self.data:
            sample_id = sample["sample_id"]
            conversation = sample["conversation"]

            # 输出完整的 conversation 结构（JSON 字符串）
            # 这样 oG-Memory 适配器可以按时间顺序逐轮写入
            conversation_json = json.dumps(conversation, ensure_ascii=False)

            corpus.append({
                "id": sample_id,
                "text": conversation_json,
                "metadata": {
                    "sample_id": sample_id,
                    "session_count": len([k for k in conversation.keys() if k.startswith("session_")]),
                }
            })

        return corpus

    def prepare_corpus_by_session(self) -> list[dict]:
        """按 session 准备语料（替代方法）

        每个 sample 的每个 session 作为一个文档。
        用于测试按 session 粒度的索引。

        Returns:
            语料列表 [{id, text, metadata}]
        """
        import json

        corpus = []

        for sample in self.data:
            sample_id = sample["sample_id"]
            conversation = sample["conversation"]

            # 找出所有 session
            session_keys = sorted([
                k for k in conversation.keys()
                if k.startswith("session_") and not k.endswith("_date_time")
            ])

            for session_key in session_keys:
                session_dialogs = conversation.get(session_key, [])
                session_date = conversation.get(f"{session_key}_date_time", "")

                # 提取 session 编号
                session_num = session_key.split("_")[1]

                # 构建 session 对话列表
                session_data = {
                    session_key: session_dialogs,
                    f"{session_key}_date_time": session_date,
                }

                corpus.append({
                    "id": f"{sample_id}_{session_key}",
                    "text": json.dumps(session_data, ensure_ascii=False),
                    "metadata": {
                        "sample_id": sample_id,
                        "session_num": session_num,
                        "date": session_date,
                        "dialog_count": len(session_dialogs),
                    }
                })

                # 记录用于后续映射
                self._corpus_ids.append(f"{sample_id}_{session_key}")
                self._corpus_timestamps.append(session_date)

        return corpus

    def prepare_queries(self) -> list[QueryEntry]:
        """准备查询

        从 LoCoMo 数据集提取问答对

        Returns:
            查询列表
        """
        queries = []
        self._query_to_sample: dict[str, str] = {}

        for sample in self.data:
            conversation = sample["conversation"]
            sample_id = sample["sample_id"]

            # 解析会话以获取证据映射
            sessions = load_conversation_sessions(conversation)
            evidence_map: dict[str, str] = {}

            for sess in sessions:
                for d in sess["dialogs"]:
                    dia_id = d.get("dia_id", f"D{sess['session_num']}:?")
                    full_id = f"{sample_id}_{dia_id}"
                    evidence_map[dia_id] = full_id

                    # 解析 D1:3 格式，映射到 session 级别
                    # 这样当 granularity=session 时，可以正确映射到 session ID
                    if ":" in dia_id:
                        try:
                            prefix = dia_id.split(":")[0]  # D1
                            num_part = prefix[1:]  # 1
                            session_ref = f"session_{num_part}"  # session_1
                            full_session_id = f"{sample_id}_{session_ref}"
                            evidence_map[f"{prefix}:{dia_id.split(':')[1]}"] = full_session_id
                        except (IndexError, ValueError):
                            pass

                # 会话级别
                session_id = f"session_{sess['session_num']}"
                full_session_id = f"{sample_id}_{session_id}"
                evidence_map[session_id] = full_session_id

            for qa in sample.get("qa", []):
                question = qa["question"]
                # adversarial 类别使用 adversarial_answer 字段
                answer = qa.get("answer") or qa.get("adversarial_answer", "")
                category = qa.get("category", 0)
                evidence_refs = qa.get("evidence", [])

                # 解析证据引用为实际文档 ID
                evidence_ids = []
                for ref in evidence_refs:
                    if ref in evidence_map:
                        evidence_ids.append(evidence_map[ref])
                    else:
                        # 尝试解析格式如 D1:3
                        resolved = resolve_evidence_ids([ref], sample)
                        evidence_ids.extend(resolved)

                query_id = f"{sample_id}_q{len(queries)}"

                queries.append(QueryEntry(
                    id=query_id,
                    question=question,
                    answer=answer,
                    evidence_ids=evidence_ids,
                    metadata={
                        "sample_id": sample_id,
                        "category": category,
                        "category_name": CATEGORIES.get(category, "Unknown"),
                    }
                ))

        return queries

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
        retrieved_ids = [r.id for r in retrieved]
        ground_truth_ids = query.evidence_ids

        # 检索指标
        retrieval_metrics = {
            f"recall@{top_k}": recall_at_k(retrieved_ids, set(ground_truth_ids), top_k),
            f"ndcg@{top_k}": ndcg_at_k(retrieved_ids, set(ground_truth_ids), top_k),
            "any_hit": 1.0 if len(set(retrieved_ids) & set(ground_truth_ids)) > 0 else 0.0,
        }

        # 如果有检索结果，尝试计算问答指标
        qa_metrics = {}
        if retrieved_ids:
            best_retrieved_text = next(
                (r.text for r in retrieved if r.id in ground_truth_ids),
                retrieved[0].text if retrieved else ""
            )
            if best_retrieved_text:
                qa_metrics = {
                    "f1": f1_score(best_retrieved_text, query.answer),
                    "exact_match": exact_match(best_retrieved_text, query.answer),
                }

        return EvaluationResult(
            query_id=query.id,
            question=query.question,
            retrieved_ids=retrieved_ids,
            ground_truth_ids=ground_truth_ids,
            retrieval_metrics=retrieval_metrics,
            qa_metrics=qa_metrics,
            metadata=query.metadata,
        )

    def compute_aggregate_metrics(self, results: list[EvaluationResult]) -> dict:
        """计算聚合指标

        Args:
            results: 评估结果列表

        Returns:
            聚合指标字典
        """
        if not results:
            return {}

        # 按类别分组
        category_results: dict[int, list[EvaluationResult]] = {}
        for r in results:
            cat = r.metadata.get("category", 0)
            if cat not in category_results:
                category_results[cat] = []
            category_results[cat].append(r)

        # 计算总体指标
        total_recall = sum(r.retrieval_metrics.get(f"recall@10", 0.0) for r in results) / len(results)
        total_ndcg = sum(r.retrieval_metrics.get(f"ndcg@10", 0.0) for r in results) / len(results)
        total_any_hit = sum(r.retrieval_metrics.get("any_hit", 0.0) for r in results) / len(results)

        # 计算类别指标
        category_metrics = {}
        for cat, cat_results in category_results.items():
            cat_name = CATEGORIES.get(cat, f"Category_{cat}")
            cat_recall = sum(r.retrieval_metrics.get(f"recall@10", 0.0) for r in cat_results) / len(cat_results)
            category_metrics[cat_name] = {
                "count": len(cat_results),
                "recall@10": cat_recall,
            }

        return {
            "total": {
                "count": len(results),
                "recall@10": total_recall,
                "ndcg@10": total_ndcg,
                "any_hit@10": total_any_hit,
            },
            "by_category": category_metrics,
        }