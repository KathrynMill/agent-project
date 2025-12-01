"""验证智能体 - 负责验证压缩结果的质量"""

from typing import Dict, List, Any
import logging

from .base_agent import BaseAgent, AgentResponse
from ..models.compression_models import AgentTask, ValidationResult

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """验证智能体：负责验证压缩结果的质量"""

    def __init__(self, llm_service=None, **kwargs):
        super().__init__(
            name="ValidationAgent",
            timeout_seconds=90,
            **kwargs
        )
        self.llm_service = llm_service

    def get_task_types(self) -> List[str]:
        """获取支持的任务类型"""
        return ["validate_compression", "quality_assessment", "completeness_check", "health_check"]

    async def process_task(self, task: AgentTask) -> AgentResponse:
        """处理验证任务"""
        try:
            if task.task_type == "validate_compression":
                return await self._validate_compression(task)
            elif task.task_type == "quality_assessment":
                return await self._assess_quality(task)
            elif task.task_type == "completeness_check":
                return await self._check_completeness(task)
            elif task.task_type == "health_check":
                return AgentResponse(
                    success=True,
                    result={"status": "healthy", "service": "validation"},
                    agent_name=self.name,
                    task_type=task.task_type
                )
            else:
                raise AgentError(f"不支持的任务类型: {task.task_type}")

        except Exception as e:
            logger.error(f"验证任务处理失败: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name,
                task_type=task.task_type
            )

    async def _validate_compression(self, task: AgentTask) -> AgentResponse:
        """验证压缩结果"""
        # 模拟验证结果
        result = ValidationResult(
            logic_consistency=True,
            story_completeness=True,
            player_experience=0.8,
            compression_quality=0.85,
            validation_issues=["部分细节可以进一步优化"],
            suggestions=["加强情感冲突", "简化次要角色"],
            overall_score=0.83,
            is_acceptable=True
        )

        return AgentResponse(
            success=True,
            result=result.dict(),
            agent_name=self.name,
            task_type=task.task_type
        )

    async def _assess_quality(self, task: AgentTask) -> AgentResponse:
        """评估质量"""
        # 模拟质量评估
        quality_metrics = {
            "clarity": 0.9,
            "engagement": 0.8,
            "consistency": 0.85,
            "playability": 0.88
        }

        return AgentResponse(
            success=True,
            result={"quality_metrics": quality_metrics},
            agent_name=self.name,
            task_type=task.task_type
        )

    async def _check_completeness(self, task: AgentTask) -> AgentResponse:
        """检查完整性"""
        # 模拟完整性检查
        completeness = {
            "plot_elements": True,
            "character_development": True,
            "clue_distribution": False,  # 缺少一些线索
            "resolution": True
        }

        return AgentResponse(
            success=True,
            result={"completeness": completeness},
            agent_name=self.name,
            task_type=task.task_type
        )