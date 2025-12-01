"""
专用压缩服务 - 替换多智能体架构
提供剧本压缩的核心业务逻辑
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import json
from pathlib import Path
import time
from dataclasses import asdict

from ..models.specialized_compression_model import (
    SpecializedCompressionModel, CompressionConfig, CompressionLevel,
    ScriptElements, CompressionMetrics
)
from ..models.compression_models import (
    CompressionRequest, CompressionResult, CompressionProgress
)
from ..models.script_models import Script
from ..data.data_pipeline import DataPipeline, TrainingExample

logger = logging.getLogger(__name__)


class CompressionService:
    """专用压缩服务 - 单一模型替代多智能体协作"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化压缩服务

        Args:
            model_path: 模型文件路径，如果为None则使用默认路径
        """
        self.model: Optional[SpecializedCompressionModel] = None
        self.model_path = model_path or "./models/specialized_compression_model.pth"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 性能统计
        self.stats = {
            'total_compressions': 0,
            'successful_compressions': 0,
            'failed_compressions': 0,
            'total_compression_time': 0.0,
            'average_compression_time': 0.0,
            'average_compression_ratio': 0.0,
            'average_quality_score': 0.0,
            'last_compression_time': None,
            'model_load_time': None,
            'last_error': None
        }

        # 缓存
        self.compression_cache = {}
        self.cache_max_size = 100

        # 压缩历史（兼容原有接口）
        self.compression_history: List[Dict[str, Any]] = []

        # 活跃任务（兼容原有接口）
        self.active_tasks: Dict[str, CompressionProgress] = {}

        logger.info(f"专用压缩服务初始化完成，设备: {self.device}")

    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        加载专用压缩模型

        Args:
            model_path: 模型路径，如果为None则使用初始化时的路径

        Returns:
            bool: 加载是否成功
        """
        try:
            load_path = model_path or self.model_path

            if not Path(load_path).exists():
                logger.error(f"模型文件不存在: {load_path}")
                return False

            start_time = time.time()
            logger.info(f"开始加载专用模型: {load_path}")

            # 加载模型
            self.model = SpecializedCompressionModel.load_model(load_path)
            self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            self.stats['model_load_time'] = load_time

            logger.info(f"专用模型加载成功，耗时: {load_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"专用模型加载失败: {e}")
            self.stats['last_error'] = str(e)
            return False

    async def compress_script(self, request: CompressionRequest) -> CompressionResult:
        """
        压缩剧本 - 使用专用模型

        Args:
            request: 压缩请求

        Returns:
            CompressionResult: 压缩结果
        """
        if self.model is None:
            if not await self.load_model():
                raise Exception("专用压缩模型加载失败")

        compression_id = f"compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            logger.info(f"开始使用专用模型压缩剧本 {request.script.id}，任务ID: {compression_id}")

            # 创建压缩进度跟踪
            progress = CompressionProgress(
                progress_id=compression_id,
                current_step="分析剧本",
                total_steps=4,
                completed_steps=0,
                progress_percentage=0.0,
                status_message="使用专用模型分析剧本"
            )
            self.active_tasks[compression_id] = progress

            # 转换脚本为文本
            script_text = self._script_to_text(request.script)

            # 计算目标压缩比例
            current_hours = request.script.metadata.estimated_duration_hours
            target_ratio = request.target_hours / current_hours

            # 选择压缩级别
            if target_ratio >= 0.7:
                compression_level = CompressionLevel.LIGHT
            elif target_ratio >= 0.5:
                compression_level = CompressionLevel.MEDIUM
            else:
                compression_level = CompressionLevel.HEAVY

            # 创建压缩配置
            config = CompressionConfig(
                target_ratio=target_ratio,
                compression_level=compression_level,
                preserve_elements=getattr(request, 'preserve_elements', [])
            )

            progress.current_step = "执行压缩"
            progress.completed_steps = 1
            progress.progress_percentage = 25.0
            progress.status_message = "专用模型正在执行智能压缩"

            # 执行压缩
            compression_result = await self._compress_with_specialized_model(
                script_text, config, use_cache=True
            )

            progress.current_step = "验证质量"
            progress.completed_steps = 2
            progress.progress_percentage = 75.0
            progress.status_message = "验证压缩质量"

            if not compression_result['success']:
                raise Exception(f"专用模型压缩失败: {compression_result.get('error', '未知错误')}")

            # 转换为CompressionResult格式（兼容原有接口）
            result = CompressionResult(
                success=True,
                compression_id=compression_id,
                compressed_script=compression_result['compressed_script'],
                compression_ratio=compression_result['metrics']['compression_ratio'],
                quality_score=compression_result['metrics']['overall_quality'],
                processing_time=compression_result['processing_time'],
                original_duration=request.script.metadata.estimated_duration_hours,
                compressed_duration=request.target_hours,
                preserved_elements=config.preserve_elements,
                metrics=compression_result['metrics']
            )

            progress.current_step = "完成"
            progress.completed_steps = 4
            progress.progress_percentage = 100.0
            progress.status_message = "专用模型压缩完成"

            # 记录历史
            history_record = {
                "compression_id": compression_id,
                "script_id": request.script.id,
                "original_duration": request.script.metadata.estimated_duration_hours,
                "target_duration": request.target_hours,
                "success": result.success,
                "compression_ratio": result.compression_ratio,
                "quality_score": result.quality_score,
                "timestamp": datetime.now().isoformat(),
                "model_type": "specialized_compression_model"
            }
            self.compression_history.append(history_record)

            logger.info(f"专用模型剧本压缩完成: {compression_id}")
            return result

        except Exception as e:
            logger.error(f"专用模型剧本压缩失败: {str(e)}")

            # 更新进度为失败
            if compression_id in self.active_tasks:
                progress = self.active_tasks[compression_id]
                progress.current_step = "失败"
                progress.status_message = f"专用模型压缩失败: {str(e)}"

            raise Exception(f"专用模型剧本压缩失败: {str(e)}")

        finally:
            # 清理活跃任务（延迟一段时间，以便查询状态）
            if compression_id in self.active_tasks:
                await asyncio.sleep(30)  # 保留30秒状态
                del self.active_tasks[compression_id]

    async def _compress_with_specialized_model(self,
                                             script_text: str,
                                             config: CompressionConfig,
                                             use_cache: bool = True) -> Dict[str, Any]:
        """使用专用模型执行压缩"""
        # 检查缓存
        if use_cache:
            cache_key = self._generate_cache_key(script_text, config)
            if cache_key in self.compression_cache:
                logger.info("使用专用模型缓存结果")
                return self.compression_cache[cache_key]

        start_time = time.time()

        try:
            # 输入验证
            validation_result = self._validate_input(script_text, config)
            if not validation_result['valid']:
                raise Exception(validation_result['error'])

            # 执行压缩
            with torch.no_grad():
                result = self.model(script_text, config)

            # 处理结果
            compression_result = {
                'compression_id': f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'success': True,
                'compressed_script': result['compressed_text'],
                'metrics': asdict(result['metrics']),
                'compression_config': asdict(config),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'script_length': len(script_text),
                'compressed_length': len(result['compressed_text'])
            }

            # 更新统计信息
            self._update_stats(compression_result, start_time)

            # 缓存结果
            if use_cache:
                self._cache_result(cache_key, compression_result)

            return compression_result

        except Exception as e:
            error_msg = f"专用模型压缩失败: {str(e)}"
            logger.error(error_msg)

            self.stats['failed_compressions'] += 1
            self.stats['last_error'] = error_msg

            return {
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _script_to_text(self, script: Script) -> str:
        """将剧本对象转换为文本"""
        text_parts = []

        # 添加剧本标题和描述
        if script.metadata.title:
            text_parts.append(f"剧本标题: {script.metadata.title}")
        if script.metadata.description:
            text_parts.append(f"剧本描述: {script.metadata.description}")

        # 添加人物介绍
        if script.player_scripts:
            text_parts.append("\n人物介绍:")
            for player_id, player_script in script.player_scripts.items():
                if hasattr(player_script, 'character') and player_script.character:
                    text_parts.append(f"{player_script.character.name}: {player_script.character.description}")

        # 添加剧本正文
        if script.master_script and hasattr(script.master_script, 'content'):
            text_parts.append(f"\n剧本正文:\n{script.master_script.content}")

        return "\n".join(text_parts)

    async def get_compression_progress(self, compression_id: str) -> Optional[CompressionProgress]:
        """
        获取压缩进度

        Args:
            compression_id: 压缩任务ID

        Returns:
            Optional[CompressionProgress]: 压缩进度
        """
        return self.active_tasks.get(compression_id)

    async def cancel_compression(self, compression_id: str) -> bool:
        """
        取消压缩任务 - 专用模型版本

        Args:
            compression_id: 压缩任务ID

        Returns:
            bool: 是否成功取消
        """
        if compression_id in self.active_tasks:
            # 专用模型的取消逻辑（简化版本）
            try:
                # 标记任务为取消状态
                progress = self.active_tasks[compression_id]
                progress.current_step = "已取消"
                progress.status_message = "任务被用户取消"

                # 清理本地任务
                del self.active_tasks[compression_id]

                logger.info(f"专用模型压缩任务已取消: {compression_id}")
                return True

            except Exception as e:
                logger.error(f"取消专用模型压缩任务失败: {str(e)}")
                return False
        else:
            return False

    async def estimate_compression(self, script: Script, target_hours: float, strategy: str = "balanced") -> Dict[str, Any]:
        """
        估算压缩效果 - 专用模型版本

        Args:
            script: 剧本对象
            target_hours: 目标时长
            strategy: 压缩策略

        Returns:
            Dict[str, Any]: 估算结果
        """
        try:
            # 转换脚本为文本
            script_text = self._script_to_text(script)

            # 使用专用模型进行估算
            estimation = await self._estimate_with_specialized_model(
                script_text, target_hours, strategy
            )

            # 添加剧本特定信息
            original_hours = script.metadata.estimated_duration_hours
            complexity_factors = {
                "player_count": script.metadata.player_count_max,
                "entity_count": len(script.entities),
                "event_count": len(script.events),
                "relation_count": len(script.relations)
            }

            # 合并估算结果
            result = {
                **estimation,
                "original_duration_hours": original_hours,
                "complexity_factors": complexity_factors,
                "estimated_time_minutes": complexity_factors.get("entity_count", 10) * 2,
                "model_type": "specialized_compression_model"
            }

            return result

        except Exception as e:
            logger.error(f"专用模型压缩估算失败: {str(e)}")
            raise Exception(f"专用模型压缩估算失败: {str(e)}")

    async def _estimate_with_specialized_model(self,
                                             script_text: str,
                                             target_hours: float,
                                             strategy: str = "balanced") -> Dict[str, Any]:
        """使用专用模型进行压缩估算"""
        try:
            # 计算当前剧本时长
            current_duration = self._estimate_duration(script_text)
            if current_duration <= 0:
                raise Exception("无法估算剧本时长")

            # 计算目标压缩比例
            target_ratio = target_hours / current_duration

            # 根据策略调整配置
            config = self._create_config_from_strategy(target_ratio, strategy)

            # 使用模型进行快速估算
            if self.model:
                result = self.model(script_text, config)
                estimated_quality = result['metrics'].overall_quality
            else:
                # 简单估算（模型未加载时）
                estimated_quality = max(0.5, 1.0 - abs(target_ratio - 0.5))

            estimation = {
                'current_duration_hours': current_duration,
                'target_duration_hours': target_hours,
                'target_compression_ratio': target_ratio,
                'estimated_compression_level': config.compression_level.value,
                'estimated_quality_score': estimated_quality,
                'strategy': strategy,
                'feasibility': self._assess_feasibility(target_ratio, estimated_quality),
                'recommendations': self._generate_recommendations(target_ratio, estimated_quality),
                'difficulty': 'easy' if target_ratio > 0.7 else 'medium' if target_ratio > 0.5 else 'hard'
            }

            return estimation

        except Exception as e:
            logger.error(f"专用模型压缩估算失败: {e}")
            raise Exception(f"专用模型压缩估算失败: {str(e)}")

    def _get_compression_recommendations(self, ratio: float, factors: Dict[str, Any]) -> List[str]:
        """获取压缩建议"""
        recommendations = []

        if ratio < 0.5:
            recommendations.append("建议优先移除非关键情节")
            recommendations.append("可以考虑合并相似角色")

        if factors.get("entity_count", 0) > 20:
            recommendations.append("重点关注人物关系简化")

        if factors.get("event_count", 0) > 15:
            recommendations.append("可以合并或删减次要事件")

        return recommendations

    async def get_compression_statistics(self) -> Dict[str, Any]:
        """
        获取压缩统计信息 - 专用模型版本

        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            # 从历史记录获取统计信息
            total_compressions = len(self.compression_history)
            successful_compressions = sum(1 for record in self.compression_history if record["success"])

            if total_compressions > 0:
                success_rate = successful_compressions / total_compressions
                avg_ratio = sum(record["compression_ratio"] for record in self.compression_history) / total_compressions
                avg_quality = sum(record["quality_score"] for record in self.compression_history) / total_compressions
            else:
                success_rate = 0.0
                avg_ratio = 0.0
                avg_quality = 0.0

            # 获取专用模型统计信息
            model_stats = self.get_compression_stats()

            return {
                "total_compressions": total_compressions,
                "successful_compressions": successful_compressions,
                "success_rate": success_rate,
                "average_compression_ratio": avg_ratio,
                "average_quality_score": avg_quality,
                "active_tasks": len(self.active_tasks),
                "model_stats": model_stats,
                "model_type": "specialized_compression_model"
            }

        except Exception as e:
            logger.error(f"获取专用模型统计信息失败: {str(e)}")
            return {"error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """获取专用模型信息"""
        if self.model is None:
            return {
                'model_loaded': False,
                'model_path': self.model_path,
                'device': str(self.device),
                'stats': self.stats
            }

        return {
            'model_loaded': True,
            'model_path': self.model_path,
            'model_name': self.model.model_name,
            'device': str(self.device),
            'hidden_size': self.model.hidden_size,
            'vocab_size': self.model.vocab_size,
            'stats': self.stats
        }

    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        stats = self.stats.copy()

        # 计算成功率
        if stats['total_compressions'] > 0:
            stats['success_rate'] = stats['successful_compressions'] / stats['total_compressions']
        else:
            stats['success_rate'] = 0.0

        # 缓存统计
        stats['cache_size'] = len(self.compression_cache)
        stats['cache_hit_rate'] = self._calculate_cache_hit_rate()

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查 - 专用模型版本

        Returns:
            Dict[str, Any]: 健康状态
        """
        try:
            # 检查专用模型状态
            model_status = self.get_model_info()

            # 检查GPU/CPU可用性
            device_status = {
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "device_memory": "Unknown"
            }

            if torch.cuda.is_available():
                device_status["device_memory"] = {
                    "total": torch.cuda.get_device_properties(0).total_memory,
                    "allocated": torch.cuda.memory_allocated(),
                    "cached": torch.cuda.memory_reserved()
                }

            # 整体状态
            overall_status = "healthy"
            if not model_status['model_loaded']:
                overall_status = "unhealthy"

            return {
                "overall_status": overall_status,
                "model_status": model_status,
                "device_status": device_status,
                "active_tasks": len(self.active_tasks),
                "total_compressions": len(self.compression_history),
                "model_type": "specialized_compression_model"
            }

        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "model_type": "specialized_compression_model"
            }

    def _validate_input(self, script_text: str, config: CompressionConfig) -> Dict[str, Any]:
        """验证输入参数"""
        if not script_text or not script_text.strip():
            return {'valid': False, 'error': '剧本文本不能为空'}

        if len(script_text) < 100:
            return {'valid': False, 'error': '剧本文本过短（至少100字符）'}

        if len(script_text) > 50000:
            return {'valid': False, 'error': '剧本文本过长（最多50000字符）'}

        if not (0.3 <= config.target_ratio <= 0.9):
            return {'valid': False, 'error': '压缩比例必须在0.3-0.9之间'}

        if not isinstance(config.compression_level, CompressionLevel):
            return {'valid': False, 'error': '无效的压缩级别'}

        return {'valid': True, 'error': None}

    def _generate_cache_key(self, script_text: str, config: CompressionConfig) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{script_text[:1000]}_{config.target_ratio}_{config.compression_level.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """缓存压缩结果"""
        if len(self.compression_cache) >= self.cache_max_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.compression_cache))
            del self.compression_cache[oldest_key]

        self.compression_cache[cache_key] = result

    def _update_stats(self, result: Dict[str, Any], start_time: float):
        """更新统计信息"""
        if result['success']:
            self.stats['successful_compressions'] += 1
            self.stats['total_compression_time'] += result['processing_time']

            # 更新平均值
            total = self.stats['successful_compressions']
            self.stats['average_compression_time'] = self.stats['total_compression_time'] / total

            # 更新质量指标
            metrics = result['metrics']
            self.stats['average_compression_ratio'] = (
                (self.stats['average_compression_ratio'] * (total - 1) + metrics['compression_ratio']) / total
            )
            self.stats['average_quality_score'] = (
                (self.stats['average_quality_score'] * (total - 1) + metrics['overall_quality']) / total
            )
        else:
            self.stats['failed_compressions'] += 1

        self.stats['total_compressions'] += 1
        self.stats['last_compression_time'] = datetime.now().isoformat()

    def _estimate_duration(self, script_text: str) -> float:
        """估算剧本游戏时长（小时）"""
        # 简单的时长估算：假设每小时阅读3000字符
        char_count = len(script_text)
        estimated_hours = char_count / 3000.0

        # 根据剧本复杂度调整
        import re
        complexity_factors = [
            len(re.findall(r'第.*章|第.*幕', script_text)) * 0.5,  # 章节数
            len(re.findall(r'说.*：|道.*：', script_text)) * 0.01,  # 对话数
            len(re.findall(r'线索|证据', script_text)) * 0.2      # 线索数
        ]

        adjusted_hours = estimated_hours * (1 + sum(complexity_factors))
        return max(0.5, adjusted_hours)  # 至少半小时

    def _create_config_from_strategy(self, target_ratio: float, strategy: str) -> CompressionConfig:
        """根据策略创建压缩配置"""
        # 根据目标比例选择压缩级别
        if target_ratio >= 0.7:
            level = CompressionLevel.LIGHT
        elif target_ratio >= 0.5:
            level = CompressionLevel.MEDIUM
        else:
            level = CompressionLevel.HEAVY

        # 根据策略调整权重
        if strategy == "preserve_logic":
            config = CompressionConfig(
                target_ratio=target_ratio,
                compression_level=level,
                logic_weight=0.5,
                story_weight=0.2,
                playability_weight=0.2,
                length_weight=0.1
            )
        elif strategy == "preserve_story":
            config = CompressionConfig(
                target_ratio=target_ratio,
                compression_level=level,
                logic_weight=0.2,
                story_weight=0.5,
                playability_weight=0.2,
                length_weight=0.1
            )
        elif strategy == "fast":
            config = CompressionConfig(
                target_ratio=target_ratio,
                compression_level=level,
                logic_weight=0.2,
                story_weight=0.2,
                playability_weight=0.3,
                length_weight=0.3
            )
        else:  # balanced
            config = CompressionConfig(
                target_ratio=target_ratio,
                compression_level=level
            )

        return config

    def _assess_feasibility(self, target_ratio: float, estimated_quality: float) -> str:
        """评估压缩可行性"""
        if target_ratio < 0.3:
            return "low"
        elif target_ratio < 0.5 and estimated_quality < 0.7:
            return "medium"
        elif target_ratio > 0.8:
            return "high"
        else:
            return "high"

    def _generate_recommendations(self, target_ratio: float, estimated_quality: float) -> List[str]:
        """生成压缩建议"""
        recommendations = []

        if target_ratio < 0.4:
            recommendations.append("重度压缩可能影响游戏体验，建议分阶段压缩")

        if estimated_quality < 0.6:
            recommendations.append("建议提高压缩质量要求，或选择保留更多关键元素")

        if target_ratio > 0.8:
            recommendations.append("轻度压缩，建议关注逻辑细节的保留")

        recommendations.append("建议在压缩后进行人工验证和微调")

        return recommendations

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        # 这里需要记录缓存命中次数，简化实现
        if not hasattr(self, '_cache_hits'):
            self._cache_hits = 0
        if not hasattr(self, '_cache_requests'):
            self._cache_requests = 0

        return self._cache_hits / max(1, self._cache_requests)