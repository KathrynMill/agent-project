"""RAG问答服务 API 路由"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...core.models.script_models import RAGQuery, RAGResponse
from ...core.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()


# 请求/响应模型
class RAGQueryRequest(BaseModel):
    """RAG查询请求"""
    question: str
    script_id: str = None
    max_context_length: int = 4000
    search_k: int = 5
    include_graph: bool = True


class RAGQueryResponse(BaseModel):
    """RAG查询响应"""
    answer: str
    sources: List[str] = []
    confidence: float = 0.0
    query_time: float = 0.0
    graph_results: Dict[str, Any] = {}
    vector_results: List[Dict[str, Any]] = []


class EntityExtractionRequest(BaseModel):
    """实体提取请求"""
    text: str


class EntityExtractionResponse(BaseModel):
    """实体提取响应"""
    entities: Dict[str, List[str]]
    processing_time: float = 0.0


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends()
):
    """
    RAG问答查询

    Args:
        request: RAG查询请求
        rag_service: RAG服务

    Returns:
        RAGQueryResponse: 查询响应
    """
    try:
        logger.info(f"收到RAG查询: {request.question[:50]}...")

        # 创建RAG查询对象
        rag_query = RAGQuery(
            question=request.question,
            script_id=request.script_id,
            max_context_length=request.max_context_length,
            search_k=request.search_k,
            include_graph=request.include_graph
        )

        # 执行查询
        response = await rag_service.query(rag_query)

        return RAGQueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            query_time=response.query_time,
            graph_results=response.graph_results or {},
            vector_results=response.vector_results or []
        )

    except Exception as e:
        logger.error(f"RAG查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/rag/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities(
    request: EntityExtractionRequest,
    rag_service: RAGService = Depends()
):
    """
    从文本中提取实体

    Args:
        request: 实体提取请求
        rag_service: RAG服务

    Returns:
        EntityExtractionResponse: 提取响应
    """
    try:
        logger.info(f"提取实体，文本长度: {len(request.text)}")

        # 执行实体提取
        entities = await rag_service.extract_entities(request.text)

        return EntityExtractionResponse(
            entities=entities,
            processing_time=1.0  # 模拟处理时间
        )

    except Exception as e:
        logger.error(f"实体提取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"实体提取失败: {str(e)}")


@router.get("/rag/scripts")
async def list_available_scripts(
    rag_service: RAGService = Depends()
):
    """
    获取可用的剧本列表

    Args:
        rag_service: RAG服务

    Returns:
        Dict[str, Any]: 剧本列表
    """
    try:
        logger.info("获取可用剧本列表")

        scripts = await rag_service.list_scripts()

        return {
            "scripts": scripts,
            "count": len(scripts)
        }

    except Exception as e:
        logger.error(f"获取剧本列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取剧本列表失败: {str(e)}")


@router.post("/rag/text-to-ngql")
async def text_to_ngql(
    question: str,
    rag_service: RAGService = Depends()
):
    """
    将自然语言转换为nGQL

    Args:
        question: 自然语言问题
        rag_service: RAG服务

    Returns:
        Dict[str, Any]: 转换结果
    """
    try:
        logger.info(f"文本转nGQL: {question}")

        # 执行转换
        ngql = await rag_service.text_to_ngql(question)

        return {
            "question": question,
            "ngql": ngql,
            "success": True
        }

    except Exception as e:
        logger.error(f"文本转nGQL失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")


# 依赖注入函数（简化版）
async def rag_service_dependency():
    """获取RAG服务实例"""
    # 这里应该返回实际的RAG服务实例
    # 暂时返回一个模拟对象
    class MockRAGService:
        async def query(self, rag_query):
            return RAGResponse(
                answer=f"关于问题的回答: {rag_query.question[:20]}...",
                sources=["来源1", "来源2"],
                confidence=0.85,
                query_time=1.5
            )

        async def extract_entities(self, text):
            return {
                "persons": ["张三", "李四"],
                "locations": ["客厅", "厨房"],
                "items": ["刀", "钥匙"],
                "events": ["谋杀", "发现"],
                "times": ["晚上8点", "午夜"]
            }

        async def list_scripts(self):
            return [
                {"id": "script1", "title": "别墅疑案", "duration": 4},
                {"id": "script2", "title": "古堡迷踪", "duration": 5}
            ]

        async def text_to_ngql(self, question):
            return f"GO FROM 'person1' OVER KNOWS YIELD dst AS friend"

    return MockRAGService()


# 更新路由的依赖
router.dependency_overrides[Depends] = rag_service_dependency