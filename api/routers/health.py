"""健康检查 API 路由"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends
from datetime import datetime

from ...core.services.compression_service import CompressionService
from ...core.services.llm_service import LLMService
from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check():
    """
    系统健康检查

    Returns:
        Dict[str, Any]: 系统健康状态
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.application.app_version,
            "services": {}
        }

        # 检查各个服务
        llm_health = await _check_llm_service()
        health_status["services"]["llm"] = llm_health

        compression_health = await _check_compression_service()
        health_status["services"]["compression"] = compression_health

        # 检查配置
        config_health = _check_configuration()
        health_status["services"]["configuration"] = config_health

        # 判断整体状态
        service_statuses = [service.get("status", "unhealthy") for service in health_status["services"].values()]
        if all(status == "healthy" for status in service_statuses):
            health_status["status"] = "healthy"
        elif any(status == "unhealthy" for status in service_statuses):
            health_status["status"] = "unhealthy"
        else:
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "services": {}
        }


@router.get("/health/detailed")
async def detailed_health_check():
    """
    详细健康检查

    Returns:
        Dict[str, Any]: 详细的系统健康状态
    """
    try:
        detailed_status = await health_check()

        # 添加更详细的信息
        detailed_status["system_info"] = {
            "python_version": "3.x",  # 可以实际获取
            "environment": "development" if settings.application.debug else "production",
            "log_level": settings.application.log_level,
            "api_host": settings.application.api_host,
            "api_port": settings.application.api_port
        }

        detailed_status["dependencies"] = {
            "nebula_graph": {
                "host": settings.database.nebula_host,
                "port": settings.database.nebula_port,
                "space": settings.database.nebula_space_name
            },
            "qdrant": {
                "host": settings.database.qdrant_host,
                "port": settings.database.qdrant_port,
                "collection": settings.database.qdrant_collection_name
            },
            "llm": {
                "model": settings.llm.gemini_model,
                "temperature": settings.llm.gemini_temperature,
                "base_url": settings.get_llm_base_url()
            },
            "embedding": {
                "host": settings.embedding.embedding_host,
                "port": settings.embedding.embedding_port,
                "model": settings.embedding.embedding_model
            }
        }

        detailed_status["performance"] = {
            "max_concurrent_requests": settings.application.max_concurrent_requests,
            "request_timeout": settings.application.request_timeout,
            "agent_timeout": settings.compression.agent_timeout,
            "max_iterations": settings.compression.max_iterations
        }

        return detailed_status

    except Exception as e:
        logger.error(f"详细健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/health/services/{service_name}")
async def service_health_check(service_name: str):
    """
    特定服务健康检查

    Args:
        service_name: 服务名称

    Returns:
        Dict[str, Any]: 服务健康状态
    """
    try:
        if service_name == "llm":
            return await _check_llm_service()
        elif service_name == "compression":
            return await _check_compression_service()
        elif service_name == "configuration":
            return _check_configuration()
        else:
            return {
                "status": "unknown",
                "message": f"未知的服务: {service_name}",
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"服务健康检查失败 ({service_name}): {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/health/readiness")
async def readiness_check():
    """
    就绪检查（Kubernetes就绪探针）

    Returns:
        Dict[str, Any]: 就绪状态
    """
    try:
        # 检查关键服务是否就绪
        llm_ready = await _check_llm_service()
        compression_ready = await _check_compression_service()
        config_ready = _check_configuration()

        is_ready = (
            llm_ready.get("status") == "healthy" and
            compression_ready.get("status") == "healthy" and
            config_ready.get("status") == "healthy"
        )

        return {
            "ready": is_ready,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "llm": llm_ready.get("status") == "healthy",
                "compression": compression_ready.get("status") == "healthy",
                "configuration": config_ready.get("status") == "healthy"
            }
        }

    except Exception as e:
        logger.error(f"就绪检查失败: {str(e)}")
        return {
            "ready": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/health/liveness")
async def liveness_check():
    """
    存活检查（Kubernetes存活探针）

    Returns:
        Dict[str, Any]: 存活状态
    """
    try:
        # 简单的存活检查，只要应用能响应就认为存活
        return {
            "alive": True,
            "timestamp": datetime.now().isoformat(),
            "uptime": "运行中"  # 可以实际计算运行时间
        }

    except Exception as e:
        logger.error(f"存活检查失败: {str(e)}")
        return {
            "alive": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# 辅助函数
async def _check_llm_service() -> Dict[str, Any]:
    """检查LLM服务"""
    try:
        llm_service = LLMService()
        health = await llm_service.health_check()
        return health
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_compression_service() -> Dict[str, Any]:
    """检查压缩服务"""
    try:
        compression_service = CompressionService()
        health = await compression_service.health_check()
        return health
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def _check_configuration() -> Dict[str, Any]:
    """检查配置"""
    try:
        # 检查必要的配置项
        config_issues = []

        if not settings.llm.gemini_api_key:
            config_issues.append("GEMINI_API_KEY 未设置")

        if settings.compression.max_iterations <= 0:
            config_issues.append("max_iterations 必须大于0")

        if settings.application.max_concurrent_requests <= 0:
            config_issues.append("max_concurrent_requests 必须大于0")

        return {
            "status": "healthy" if not config_issues else "degraded",
            "issues": config_issues,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }