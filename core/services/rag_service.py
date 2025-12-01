"""RAG服务 - 检索增强生成服务"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

from ..models.script_models import RAGQuery, RAGResponse
from ..models.script_models import ScriptEntity, ScriptRelation
from .llm_service import LLMService
from .graph_service import NebulaGraphService
from .vector_service import QdrantService

from ...shared.config.settings import get_settings
from ...shared.utils.exceptions import RAGServiceError

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGService:
    """检索增强生成服务"""

    def __init__(self):
        """初始化RAG服务"""
        self.settings = settings
        self.llm_service = None
        self.graph_service = None
        self.vector_service = None
        self._initialized = False

    async def initialize(self):
        """初始化服务"""
        try:
            self.llm_service = LLMService()
            self.graph_service = NebulaGraphService()
            self.vector_service = QdrantService()

            # 初始化图数据库和向量数据库
            await self.graph_service.initialize()
            await self.vector_service.initialize()

            self._initialized = True
            logger.info("RAG服务初始化成功")

        except Exception as e:
            logger.error(f"RAG服务初始化失败: {str(e)}")
            raise RAGServiceError(f"初始化失败: {str(e)}")

    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        执行RAG查询

        Args:
            rag_query: RAG查询对象

        Returns:
            RAGResponse: 查询响应
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(f"执行RAG查询: {rag_query.question[:50]}...")
            start_time = time.time()

            # 1. 自然语言转nGQL
            graph_results = {}
            vector_results = []

            if rag_query.include_graph:
                # 2. 图谱检索
                ngql = await self.llm_service.text_to_ngql(
                    rag_query.question,
                    self._get_schema_info()
                )
                graph_results = await self._execute_graph_query(ngql)

            # 3. 向量检索
            if rag_query.search_k > 0:
                query_embedding = await self._get_query_embedding(rag_query.question)
                vector_results = await self.vector_service.search_vectors(
                    query_embedding,
                    limit=rag_query.search_k
                )

            # 4. 生成回答
            answer = await self._generate_answer(
                rag_query.question,
                graph_results,
                vector_results,
                rag_query.max_context_length
            )

            # 5. 构建响应
            processing_time = time.time() - start_time
            sources = self._extract_sources(graph_results, vector_results)

            response = RAGResponse(
                answer=answer,
                sources=sources,
                confidence=self._calculate_confidence(graph_results, vector_results),
                query_time=processing_time,
                graph_results=graph_results if graph_results else None,
                vector_results=vector_results if vector_results else None
            )

            logger.info(f"RAG查询完成，耗时 {processing_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"RAG查询失败: {str(e)}")
            raise RAGServiceError(f"查询失败: {str(e)}")

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        从文本中提取实体

        Args:
            text: 输入文本

        Returns:
            Dict[str, List[str]]: 提取的实体
        """
        if not self._initialized:
            await self.initialize()

        try:
            entities = await self.llm_service.extract_entities(text)
            return entities

        except Exception as e:
            logger.error(f"实体提取失败: {str(e)}")
            raise RAGServiceError(f"实体提取失败: {str(e)}")

    async def extract_relations(self, entities: Dict[str, List[str]], text: str) -> List[Dict[str, Any]]:
        """
        提取实体关系

        Args:
            entities: 实体字典
            text: 输入文本

        Returns:
            List[Dict[str, Any]]: 关系列表
        """
        if not self._initialized:
            await self.initialize()

        try:
            relations = await self.llm_service.extract_relations(entities, text)
            return relations

        except Exception as e:
            logger.error(f"关系提取失败: {str(e)}")
            raise RAGServiceError(f"关系提取失败: {str(e)}")

    async def text_to_ngql(self, question: str) -> str:
        """
        自然语言转nGQL

        Args:
            question: 自然语言问题

        Returns:
            str: nGQL查询语句
        """
        if not self._initialized:
            await self.initialize()

        try:
            schema_info = self._get_schema_info()
            ngql = await self.llm_service.text_to_ngql(question, schema_info)
            return ngql

        except Exception as e:
            logger.error(f"文本转nGQL失败: {str(e)}")
            raise RAGServiceError(f"转换失败: {str(e)}")

    async def list_scripts(self) -> List[Dict[str, Any]]:
        """
        获取可用剧本列表

        Returns:
            List[Dict[str, Any]]: 剧本列表
        """
        if not self._initialized:
            await self.initialize()

        try:
            # 这里应该从数据库获取剧本列表
            # 暂时返回模拟数据
            scripts = [
                {"id": "script1", "title": "别墅疑案", "duration": 4, "type": "悬疑"},
                {"id": "script2", "title": "古堡迷踪", "duration": 5, "type": "恐怖"},
                {"id": "script3", "title": "校园密案", "duration": 3, "type": "校园"}
            ]
            return scripts

        except Exception as e:
            logger.error(f"获取剧本列表失败: {str(e)}")
            raise RAGServiceError(f"获取剧本列表失败: {str(e)}")

    # 私有方法
    def _get_schema_info(self) -> str:
        """获取图谱结构信息"""
        return """
        可用的标签类型：
        - Person(name string, role string)
        - Location(name string)
        - Item(name string, type string)
        - Event(name string, description string)
        - Timeline(name string, start string, end string)

        可用的边类型：
        - LOCATED_IN, HAS_ITEM, PARTICIPATED_IN, HAPPENED_AT, HAPPENED_ON, KNOWS, KILLED, SUSPECTS
        """

    async def _execute_graph_query(self, ngql: str) -> Dict[str, Any]:
        """执行图谱查询"""
        try:
            result = await self.graph_service.execute_query(ngql)
            return self.graph_service.result_to_dict(result)
        except Exception as e:
            logger.warning(f"图谱查询失败: {str(e)}")
            return {}

    async def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的向量表示"""
        try:
            # 这里应该调用嵌入服务获取向量
            # 暂时返回模拟向量
            import random
            return [random.random() for _ in range(settings.embedding.embedding_dimension)]
        except Exception as e:
            logger.error(f"获取查询向量失败: {str(e)}")
            raise RAGServiceError(f"获取查询向量失败: {str(e)}")

    async def _generate_answer(
        self,
        question: str,
        graph_results: Dict[str, Any],
        vector_results: List[Dict[str, Any]],
        max_context_length: int
    ) -> str:
        """生成回答"""
        try:
            # 构建上下文
            context = self._build_context(graph_results, vector_results, max_context_length)

            # 构建提示
            prompt = f"""
            基于以下信息回答问题：

            问题：{question}

            相关信息：
            {context}

            请基于提供的信息给出准确、详细的回答。如果信息不足，请说明。
            """

            # 调用LLM生成回答
            from .llm_service import LLMRequest
            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7,
                system_prompt="你是一个专业的剧本分析助手，请基于提供的信息准确回答问题。"
            )

            response = await self.llm_service.generate_text(llm_request)
            return response.content

        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return f"抱歉，无法生成回答：{str(e)}"

    def _build_context(
        self,
        graph_results: Dict[str, Any],
        vector_results: List[Dict[str, Any]],
        max_length: int
    ) -> str:
        """构建上下文"""
        context_parts = []

        # 添加图谱结果
        if graph_results:
            context_parts.append("知识图谱信息：")
            for item in graph_results[:5]:  # 限制条数
                context_parts.append(f"- {str(item)}")

        # 添加向量检索结果
        if vector_results:
            context_parts.append("\n相关文本片段：")
            for item in vector_results[:3]:  # 限制条数
                if item.get("payload", {}).get("text"):
                    context_parts.append(f"- {item['payload']['text']}")

        context = "\n".join(context_parts)

        # 如果上下文过长，截断
        if len(context) > max_length:
            context = context[:max_length] + "\n...(内容已截断)"

        return context

    def _extract_sources(
        self,
        graph_results: Dict[str, Any],
        vector_results: List[Dict[str, Any]]
    ) -> List[str]:
        """提取来源信息"""
        sources = []

        if graph_results:
            sources.append("知识图谱检索")

        if vector_results:
            sources.append("向量语义检索")

        return sources

    def _calculate_confidence(
        self,
        graph_results: Dict[str, Any],
        vector_results: List[Dict[str, Any]]
    ) -> float:
        """计算置信度"""
        confidence = 0.0

        # 基于图谱结果数量计算置信度
        if graph_results:
            confidence += min(0.4, len(graph_results) * 0.1)

        # 基于向量检索结果计算置信度
        if vector_results:
            avg_score = sum(r.get("score", 0) for r in vector_results) / len(vector_results)
            confidence += avg_score * 0.6

        return min(confidence, 1.0)

    async def ping(self) -> bool:
        """检查服务状态"""
        try:
            if not self._initialized:
                await self.initialize()

            # 检查各个子服务
            llm_health = await self.llm_service.health_check()
            graph_healthy = await self.graph_service.ping()
            vector_healthy = await self.vector_service.ping()

            return (
                llm_health.get("status") == "healthy" and
                graph_healthy and
                vector_healthy
            )
        except Exception as e:
            logger.error(f"RAG服务ping失败: {str(e)}")
            return False

    async def close(self):
        """关闭服务"""
        try:
            if self.graph_service:
                await self.graph_service.close()
            if self.vector_service:
                await self.vector_service.close()
            self._initialized = False
            logger.info("RAG服务已关闭")
        except Exception as e:
            logger.error(f"关闭RAG服务失败: {str(e)}")


# 导入time模块
import time