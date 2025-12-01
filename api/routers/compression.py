"""压缩服务 API 路由"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from pydantic import BaseModel

from ...core.models.compression_models import (
    CompressionRequest, CompressionResult, CompressionProgress
)
from ...core.models.script_models import Script
from ...core.services.compression_service import CompressionService
from ...shared.utils.exceptions import CompressionError, ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()


# 请求/响应模型
class CompressScriptRequest(BaseModel):
    """压缩剧本请求"""
    script_id: str
    target_hours: int
    compression_level: str = "medium"
    strategy: str = "balanced"
    preserve_elements: list = []
    remove_elements: list = []
    custom_requirements: dict = {}


class CompressScriptResponse(BaseModel):
    """压缩剧本响应"""
    success: bool
    compression_id: Optional[str] = None
    message: str
    result: Optional[CompressionResult] = None
    estimated_time: Optional[int] = None


class CompressionStatusResponse(BaseModel):
    """压缩状态响应"""
    compression_id: str
    status: str
    progress: CompressionProgress
    result: Optional[CompressionResult] = None


# 依赖注入
async def get_compression_service(request: Request) -> CompressionService:
    """获取压缩服务实例"""
    if not hasattr(request.app.state, 'compression_service'):
        raise HTTPException(status_code=503, detail="压缩服务未初始化")
    return request.app.state.compression_service


@router.post("/compression/compress", response_model=CompressScriptResponse)
async def compress_script(
    request: CompressScriptRequest,
    background_tasks: BackgroundTasks,
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    压缩剧本 - 使用专用AI模型

    Args:
        request: 压缩请求
        background_tasks: 后台任务
        compression_service: 压缩服务

    Returns:
        CompressScriptResponse: 压缩响应
    """
    try:
        logger.info(f"收到专用模型压缩请求: 剧本ID={request.script_id}, 目标时长={request.target_hours}小时, 策略={request.strategy}")

        # 验证请求参数
        if request.target_hours < 1 or request.target_hours > 12:
            raise ValidationError("目标时长必须在1-12小时之间")

        if request.compression_level not in ["light", "medium", "heavy", "custom"]:
            raise ValidationError("压缩级别必须是: light, medium, heavy, custom")

        if request.strategy not in ["balanced", "preserve_logic", "preserve_story", "fast"]:
            raise ValidationError("压缩策略必须是: balanced, preserve_logic, preserve_story, fast")

        # 加载剧本
        script = await _load_script_from_database(request.script_id)

        # 估算压缩效果
        logger.info("使用专用模型进行压缩效果估算...")
        estimation = await compression_service.estimate_compression(
            script, request.target_hours, request.strategy
        )

        # 创建压缩请求
        compression_request = CompressionRequest(
            script=script,
            target_hours=request.target_hours,
            compression_level=request.compression_level,
            strategy=request.strategy,
            preserve_elements=request.preserve_elements,
            remove_elements=request.remove_elements,
            custom_requirements=request.custom_requirements
        )

        # 使用专用模型执行压缩
        logger.info("开始使用专用模型执行压缩...")
        result = await compression_service.compress_script(compression_request)

        if not result.success:
            raise CompressionError(f"专用模型压缩失败: {getattr(result, 'error', '未知错误')}")

        # 返回压缩结果
        return CompressScriptResponse(
            success=True,
            compression_id=result.compression_id,
            message="专用模型压缩完成",
            result=result,
            estimated_time=estimation.get("estimated_time_minutes", 30)
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except CompressionError as e:
        logger.error(f"专用模型压缩失败: {str(e)}")
        raise HTTPException(status_code=422, detail=f"压缩失败: {str(e)}")
    except Exception as e:
        logger.error(f"压缩请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"压缩请求失败: {str(e)}")


@router.get("/compression/status/{compression_id}", response_model=CompressionStatusResponse)
async def get_compression_status(
    compression_id: str,
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    获取压缩进度 - 专用模型版本

    Args:
        compression_id: 压缩任务ID
        compression_service: 压缩服务

    Returns:
        CompressionStatusResponse: 压缩状态
    """
    try:
        logger.info(f"查询专用模型压缩进度: {compression_id}")

        progress = await compression_service.get_compression_progress(compression_id)

        if progress is None:
            # 返回专用模型的模拟进度
            mock_progress = CompressionProgress(
                progress_id=compression_id,
                current_step="模型推理中",
                total_steps=4,  # 专用模型步骤更少
                completed_steps=1,
                progress_percentage=25.0,
                status_message="专用模型正在进行智能压缩...",
                current_agent="SpecializedCompressionModel"
            )
            return CompressionStatusResponse(
                compression_id=compression_id,
                status="not_found",
                progress=mock_progress
            )

        return CompressionStatusResponse(
            compression_id=compression_id,
            status="in_progress" if progress.progress_percentage < 100 else "completed",
            progress=progress
        )

    except Exception as e:
        logger.error(f"查询专用模型压缩进度失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询进度失败: {str(e)}")


@router.delete("/compression/cancel/{compression_id}")
async def cancel_compression(
    compression_id: str,
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    取消压缩任务

    Args:
        compression_id: 压缩任务ID
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 取消结果
    """
    try:
        logger.info(f"取消压缩任务: {compression_id}")

        success = await compression_service.cancel_compression(compression_id)

        return {
            "success": success,
            "message": "压缩任务已取消" if success else "任务不存在或无法取消",
            "compression_id": compression_id
        }

    except Exception as e:
        logger.error(f"取消压缩任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@router.get("/compression/estimate/{script_id}")
async def estimate_compression(
    script_id: str,
    target_hours: int,
    strategy: str = "balanced",
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    估算压缩效果 - 专用模型版本

    Args:
        script_id: 剧本ID
        target_hours: 目标时长
        strategy: 压缩策略
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 估算结果
    """
    try:
        logger.info(f"使用专用模型估算压缩效果: 剧本ID={script_id}, 目标时长={target_hours}小时, 策略={strategy}")

        if target_hours < 1 or target_hours > 12:
            raise ValidationError("目标时长必须在1-12小时之间")

        if strategy not in ["balanced", "preserve_logic", "preserve_story", "fast"]:
            raise ValidationError("压缩策略必须是: balanced, preserve_logic, preserve_story, fast")

        # 加载剧本
        script = await _load_script_from_database(script_id)

        # 使用专用模型估算压缩效果
        estimation = await compression_service.estimate_compression(script, target_hours, strategy)

        return {
            "script_id": script_id,
            "target_hours": target_hours,
            "original_hours": script.metadata.estimated_duration_hours,
            "strategy": strategy,
            "compression_ratio": estimation["target_compression_ratio"],
            "estimated_quality": estimation["estimated_quality_score"],
            "estimated_compression_level": estimation["estimated_compression_level"],
            "estimated_time_minutes": estimation.get("estimated_time_minutes", 30),
            "difficulty": estimation["difficulty"],
            "feasibility": estimation["feasibility"],
            "recommendations": estimation["recommendations"],
            "model_type": estimation.get("model_type", "specialized_compression_model")
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"专用模型压缩估算失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"估算失败: {str(e)}")


@router.get("/compression/statistics")
async def get_compression_statistics(
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    获取压缩统计信息 - 专用模型版本

    Args:
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 统计信息
    """
    try:
        logger.info("获取专用模型压缩统计信息")

        stats = await compression_service.get_compression_statistics()

        return stats

    except Exception as e:
        logger.error(f"获取专用模型统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/compression/model/info")
async def get_model_info(
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    获取专用模型信息

    Args:
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 模型信息
    """
    try:
        logger.info("获取专用压缩模型信息")

        model_info = compression_service.get_model_info()

        return model_info

    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@router.post("/compression/model/load")
async def load_model(
    model_path: Optional[str] = None,
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    加载专用模型

    Args:
        model_path: 模型文件路径
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 加载结果
    """
    try:
        logger.info(f"加载专用模型: {model_path or '默认路径'}")

        success = await compression_service.load_model(model_path)

        return {
            "success": success,
            "message": "专用模型加载成功" if success else "专用模型加载失败",
            "model_path": model_path or "默认路径",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"加载专用模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@router.post("/compression/model/analyze")
async def analyze_script_with_model(
    script_id: str,
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    使用专用模型分析剧本

    Args:
        script_id: 剧本ID
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 分析结果
    """
    try:
        logger.info(f"使用专用模型分析剧本: {script_id}")

        # 加载剧本
        script = await _load_script_from_database(script_id)

        # 转换为文本
        script_text = compression_service._script_to_text(script)

        # 使用专用模型分析
        analysis_result = await compression_service.analyze_script(script_text)

        return {
            "script_id": script_id,
            "analysis_result": analysis_result,
            "model_type": "specialized_compression_model",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"专用模型剧本分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"剧本分析失败: {str(e)}")


@router.get("/compression/model/performance")
async def get_model_performance(
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    获取专用模型性能指标

    Args:
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 性能指标
    """
    try:
        logger.info("获取专用模型性能指标")

        # 获取统计信息
        stats = compression_service.get_compression_stats()

        # 获取模型信息
        model_info = compression_service.get_model_info()

        performance_metrics = {
            "model_info": model_info,
            "performance_stats": stats,
            "health_status": await compression_service.health_check()
        }

        return performance_metrics

    except Exception as e:
        logger.error(f"获取模型性能指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")


@router.post("/compression/model/reset-stats")
async def reset_model_stats(
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    重置专用模型统计信息

    Args:
        compression_service: 压缩服务

    Returns:
        Dict[str, Any]: 重置结果
    """
    try:
        logger.info("重置专用模型统计信息")

        compression_service.reset_stats()

        return {
            "success": True,
            "message": "专用模型统计信息已重置",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"重置统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重置统计信息失败: {str(e)}")


# 辅助函数
async def _load_script_from_database(script_id: str) -> Script:
    """
    从数据库加载剧本（模拟实现）

    Args:
        script_id: 剧本ID

    Returns:
        Script: 剧本对象
    """
    # 这里应该实现从数据库加载剧本的逻辑
    # 暂时返回一个模拟剧本
    from ...core.models.script_models import (
        ScriptMetadata, PlayerScript, MasterScript, Script
    )
    from datetime import datetime

    metadata = ScriptMetadata(
        title=f"剧本_{script_id}",
        author="系统模拟",
        genre="悬疑",
        difficulty="中等",
        player_count_min=4,
        player_count_max=6,
        estimated_duration_hours=6.0
    )

    # 创建模拟的玩家剧本
    player_scripts = {}
    for i in range(1, 5):
        player_script = PlayerScript(
            player_id=f"player_{i}",
            player_name=f"角色{i}",
            content=f"这是角色{i}的剧本内容...",
            background=f"角色{i}的背景故事",
            objectives=[f"目标{i}.1", f"目标{i}.2"],
            clues=[f"线索{i}.1", f"线索{i}.2"],
            secrets=[f"秘密{i}"]
        )
        player_scripts[f"player_{i}"] = player_script

    master_script = MasterScript(
        content="主持人剧本内容...",
        timeline="完整时间线描述...",
        truth="案件真相描述...",
        key_clues=["关键线索1", "关键线索2"],
        red_herrings=["干扰信息1", "干扰信息2"],
        solution_steps=["解答步骤1", "解答步骤2"]
    )

    script = Script(
        id=script_id,
        metadata=metadata,
        player_scripts=player_scripts,
        master_script=master_script,
        entities=[],
        relations=[],
        events=[],
        timelines=[]
    )

    return script