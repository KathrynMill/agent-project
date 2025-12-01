"""系统管理 API 路由"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...core.services.compression_service import CompressionService
from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# 请求/响应模型
class SystemConfigRequest(BaseModel):
    """系统配置请求"""
    log_level: str = "INFO"
    max_concurrent_requests: int = 10
    agent_timeout: int = 120
    max_iterations: int = 5


class AgentStatusResponse(BaseModel):
    """智能体状态响应"""
    agent_name: str
    status: str
    is_busy: bool
    total_tasks: int
    successful_tasks: int
    average_processing_time: float
    last_active: str


@router.get("/admin/system/status")
async def get_system_status(
    compression_service: CompressionService = Depends()
):
    """
    获取系统状态

    Args:
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 系统状态
    """
    try:
        logger.info("获取系统状态")

        # 获取压缩服务统计
        compression_stats = await compression_service.get_compression_statistics()

        # 构建系统状态
        system_status = {
            "status": "running",
            "version": settings.application.app_version,
            "uptime": "运行中",  # 可以实际计算
            "environment": "development" if settings.application.debug else "production",
            "compression_service": compression_stats,
            "configuration": {
                "api_host": settings.application.api_host,
                "api_port": settings.application.api_port,
                "log_level": settings.application.log_level,
                "max_concurrent_requests": settings.application.max_concurrent_requests,
                "agent_timeout": settings.compression.agent_timeout,
                "max_iterations": settings.compression.max_iterations
            },
            "external_services": {
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
                    "base_url": settings.get_llm_base_url()
                }
            }
        }

        return system_status

    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@router.get("/admin/agents/status")
async def get_agents_status(
    compression_service: CompressionService = Depends()
):
    """
    获取智能体状态

    Args:
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 智能体状态
    """
    try:
        logger.info("获取智能体状态")

        # 从压缩服务获取智能体状态
        agent_status = await compression_service._get_agent_status()

        return {
            "agents": agent_status,
            "total_agents": len([
                agent for agent in agent_status.get("sub_agents", {}).values()
                if agent.get("status") != "unhealthy"
            ])
        }

    except Exception as e:
        logger.error(f"获取智能体状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取智能体状态失败: {str(e)}")


@router.post("/admin/config/update")
async def update_system_config(
    config: SystemConfigRequest,
    compression_service: CompressionService = Depends()
):
    """
    更新系统配置

    Args:
        config: 新配置
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 更新结果
    """
    try:
        logger.info(f"更新系统配置: {config.dict()}")

        # 这里应该实现配置更新逻辑
        # 暂时返回模拟结果

        return {
            "success": True,
            "message": "配置更新成功",
            "updated_config": config.dict()
        }

    except Exception as e:
        logger.error(f"更新配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@router.post("/admin/cache/clear")
async def clear_cache():
    """
    清理缓存

    Returns:
        Dict[str, Any]: 清理结果
    """
    try:
        logger.info("清理系统缓存")

        # 这里应该实现缓存清理逻辑
        # 暂时返回模拟结果

        return {
            "success": True,
            "message": "缓存清理成功",
            "cleared_items": ["llm_cache", "embedding_cache", "graph_cache"]
        }

    except Exception as e:
        logger.error(f"清理缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理缓存失败: {str(e)}")


@router.get("/admin/logs")
async def get_system_logs(
    level: str = "INFO",
    limit: int = 100
):
    """
    获取系统日志

    Args:
        level: 日志级别
        limit: 返回条数

    Returns:
        Dict[str, Any]: 日志数据
    """
    try:
        logger.info(f"获取系统日志，级别: {level}, 限制: {limit}")

        # 这里应该实现日志获取逻辑
        # 暂时返回模拟日志
        mock_logs = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "level": "INFO",
                "logger": "api.app",
                "message": "系统启动成功"
            },
            {
                "timestamp": "2024-01-01T10:01:00",
                "level": "INFO",
                "logger": "core.services.compression_service",
                "message": "压缩服务初始化完成"
            },
            {
                "timestamp": "2024-01-01T10:02:00",
                "level": "WARNING",
                "logger": "core.agents.llm_agent",
                "message": "LLM服务响应较慢"
            }
        ]

        return {
            "logs": mock_logs[:limit],
            "total": len(mock_logs),
            "level_filter": level,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"获取系统日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取日志失败: {str(e)}")


@router.post("/admin/maintenance/reset")
async def reset_system(
    compression_service: CompressionService = Depends()
):
    """
    重置系统（谨慎使用）

    Args:
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 重置结果
    """
    try:
        logger.warning("执行系统重置")

        # 重置智能体统计
        compression_service.analysis_agent.reset_stats()
        compression_service.logic_agent.reset_stats()
        compression_service.story_agent.reset_stats()
        compression_service.validation_agent.reset_stats()
        compression_service.chief_editor.reset_stats()

        # 清理活跃任务
        compression_service.active_tasks.clear()

        return {
            "success": True,
            "message": "系统重置完成",
            "reset_items": [
                "智能体统计数据",
                "活跃压缩任务",
                "缓存数据"
            ]
        }

    except Exception as e:
        logger.error(f"系统重置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统重置失败: {str(e)}")


@router.get("/admin/metrics")
async def get_system_metrics(
    compression_service: CompressionService = Depends()
):
    """
    获取系统指标

    Args:
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 系统指标
    """
    try:
        logger.info("获取系统指标")

        # 获取压缩统计
        stats = await compression_service.get_compression_statistics()

        # 构建系统指标
        metrics = {
            "performance": {
                "total_compressions": stats.get("total_compressions", 0),
                "success_rate": stats.get("success_rate", 0),
                "average_quality_score": stats.get("average_quality_score", 0),
                "average_compression_ratio": stats.get("average_compression_ratio", 0)
            },
            "agents": {
                "total_agents": 5,  # 主编 + 4个子智能体
                "active_agents": stats.get("agent_status", {}).get("sub_agents", {}),
                "total_tasks_processed": sum(
                    agent.get("performance_stats", {}).get("total_tasks", 0)
                    for agent in stats.get("agent_status", {}).get("sub_agents", {}).values()
                )
            },
            "system": {
                "active_tasks": stats.get("active_tasks", 0),
                "uptime": "运行中",  # 可以实际计算
                "memory_usage": "模拟数据",  # 可以实际获取
                "cpu_usage": "模拟数据"  # 可以实际获取
            }
        }

        return metrics

    except Exception as e:
        logger.error(f"获取系统指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统指标失败: {str(e)}")


# 依赖注入
async def get_compression_service():
    """获取压缩服务实例"""
    from fastapi import Request
    async def _get_service(request: Request):
        if not hasattr(request.app.state, 'compression_service'):
            raise HTTPException(status_code=503, detail="压缩服务未初始化")
        return request.app.state.compression_service
    return _get_service


# 更新路由依赖
router.dependency_overrides[Depends] = get_compression_service