"""专用压缩服务 API 路由"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel

from ...core.services.specialized_compression_service import (
    get_specialized_compression_service,
    compress_script_specialized
)

logger = logging.getLogger(__name__)
router = APIRouter()


# 请求/响应模型
class SpecializedCompressionRequest(BaseModel):
    """专用压缩请求"""
    script_content: str
    target_ratio: float = 0.6
    compression_level: str = "medium"  # heavy, medium, light, minimal
    preserve_elements: List[str] = []
    script_id: Optional[str] = None


class SpecializedCompressionResponse(BaseModel):
    """专用压缩响应"""
    success: bool
    script_id: Optional[str] = None
    original_length: int
    compressed_length: int
    target_ratio: float
    actual_ratio: float
    compression_level: str
    compressed_text: Optional[str] = None
    quality_scores: Dict[str, float]
    preserved_elements: List[str]
    compression_statistics: Dict[str, Any]
    processing_time_seconds: float
    model_info: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    model_name: str
    model_path: str
    training_complete: bool
    performance_metrics: Dict[str, Any]
    supported_compression_levels: List[str]
    recommended_ratios: Dict[str, float]


@router.post("/compress", response_model=SpecializedCompressionResponse)
async def compress_script_specialized_endpoint(
    request: SpecializedCompressionRequest,
    background_tasks: BackgroundTasks
):
    """
    使用专用模型压缩剧本

    支持多种压缩级别和质量评分
    """
    try:
        logger.info(f"收到专用压缩请求: 压缩级别={request.compression_level}, 目标比例={request.target_ratio}")

        # 验证请求参数
        if not request.script_content or len(request.script_content.strip()) < 100:
            raise HTTPException(status_code=400, detail="剧本内容不能为空且长度不少于100字符")

        if request.compression_level not in ["heavy", "medium", "light", "minimal"]:
            raise HTTPException(status_code=400, detail="压缩级别必须是: heavy, medium, light, minimal")

        if not 0.1 <= request.target_ratio <= 0.95:
            raise HTTPException(status_code=400, detail="目标压缩比例必须在0.1到0.95之间")

        # 构建压缩配置
        compression_config = {
            'target_ratio': request.target_ratio,
            'compression_level': request.compression_level,
            'preserve_elements': request.preserve_elements
        }

        # 调用专用压缩服务
        service = get_specialized_compression_service()
        result = await service.compress_script(request.script_content, compression_config)

        if not result['success']:
            raise HTTPException(status_code=500, detail=f"压缩失败: {result.get('error', 'Unknown error')}")

        # 构建响应
        response = SpecializedCompressionResponse(
            script_id=request.script_id,
            **result
        )

        logger.info(f"专用压缩完成: {result['actual_ratio']:.3f}压缩比, 质量{result['quality_scores']['overall_quality']:.3f}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"专用压缩过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"专用压缩失败: {str(e)}")


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    获取专用模型信息
    """
    try:
        service = get_specialized_compression_service()
        model_info = service.get_model_info()

        return ModelInfoResponse(**model_info)

    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@router.post("/batch-compress")
async def batch_compress_scripts(requests: List[SpecializedCompressionRequest]):
    """
    批量压缩剧本
    """
    try:
        if len(requests) > 10:
            raise HTTPException(status_code=400, detail="批量压缩最多支持10个剧本")

        logger.info(f"收到批量压缩请求: {len(requests)}个剧本")

        service = get_specialized_compression_service()

        # 转换请求格式
        script_data_list = []
        for i, req in enumerate(requests):
            script_data_list.append({
                'id': req.script_id or f'script_{i+1}',
                'content': req.script_content,
                'compression_config': {
                    'target_ratio': req.target_ratio,
                    'compression_level': req.compression_level,
                    'preserve_elements': req.preserve_elements
                }
            })

        # 执行批量压缩
        results = await service.batch_compress(script_data_list)

        # 转换为响应格式
        responses = []
        for result in results:
            response = SpecializedCompressionResponse(**result)
            responses.append(response)

        logger.info(f"批量压缩完成: {len(responses)}个结果")
        return responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量压缩过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量压缩失败: {str(e)}")


@router.get("/compression-levels")
async def get_compression_levels():
    """
    获取支持的压缩级别信息
    """
    return {
        "compression_levels": {
            "heavy": {
                "description": "重度压缩 - 仅保留核心情节和关键线索",
                "recommended_ratio": 0.3,
                "expected_quality": "高",
                "use_case": "快速预览、摘要生成"
            },
            "medium": {
                "description": "中度压缩 - 保留主要情节和角色关系",
                "recommended_ratio": 0.6,
                "expected_quality": "中高",
                "use_case": "标准压缩、平衡性能"
            },
            "light": {
                "description": "轻度压缩 - 保留大部分细节",
                "recommended_ratio": 0.8,
                "expected_quality": "高",
                "use_case": "细节保留、高质量需求"
            },
            "minimal": {
                "description": "最小压缩 - 仅清理格式",
                "recommended_ratio": 0.95,
                "expected_quality": "极高",
                "use_case": "格式优化、微调"
            }
        },
        "quality_metrics": {
            "overall_quality": "整体质量评分 (0-1)",
            "compression_ratio_score": "压缩比例评分",
            "preservation_score": "元素保留评分",
            "readability_score": "可读性评分",
            "playability_score": "可玩性评分"
        }
    }


@router.post("/test-compression")
async def test_compression_quality(script_content: str):
    """
    测试不同压缩级别的效果
    """
    try:
        if len(script_content) < 100:
            raise HTTPException(status_code=400, detail="剧本内容长度不少于100字符")

        logger.info("开始压缩质量测试")

        # 测试所有压缩级别
        test_levels = ["heavy", "medium", "light", "minimal"]
        test_results = {}

        service = get_specialized_compression_service()

        for level in test_levels:
            # 使用推荐的压缩比例
            recommended_ratios = {"heavy": 0.3, "medium": 0.6, "light": 0.8, "minimal": 0.95}
            target_ratio = recommended_ratios[level]

            compression_config = {
                'target_ratio': target_ratio,
                'compression_level': level,
                'preserve_elements': []
            }

            result = await service.compress_script(script_content, compression_config)

            test_results[level] = {
                "target_ratio": target_ratio,
                "actual_ratio": result['actual_ratio'],
                "compressed_length": result['compressed_length'],
                "quality_scores": result['quality_scores'],
                "preserved_elements": result['preserved_elements'],
                "processing_time": result['processing_time_seconds']
            }

        logger.info("压缩质量测试完成")
        return {
            "script_length": len(script_content),
            "test_results": test_results,
            "recommendation": _get_best_recommendation(test_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"压缩质量测试失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"压缩质量测试失败: {str(e)}")


def _get_best_recommendation(test_results: Dict[str, Any]) -> Dict[str, Any]:
    """根据测试结果给出最佳推荐"""
    best_level = "medium"
    best_score = 0

    for level, result in test_results.items():
        # 综合评分：质量 + 压缩效果
        quality_score = result['quality_scores']['overall_quality']
        compression_efficiency = 1 - result['actual_ratio']
        combined_score = (quality_score * 0.7) + (compression_efficiency * 0.3)

        if combined_score > best_score:
            best_score = combined_score
            best_level = level

    return {
        "recommended_level": best_level,
        "recommended_ratio": test_results[best_level]['actual_ratio'],
        "expected_quality": test_results[best_level]['quality_scores']['overall_quality'],
        "reasoning": f"基于质量评分({best_score:.3f})和压缩效率的综合考量"
    }