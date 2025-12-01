"""故事智能体 - 负责保持故事性的压缩"""

from typing import Dict, List, Any
import logging

from .base_agent import BaseAgent, AgentResponse
from ..models.compression_models import AgentTask, StoryCompressionResult

logger = logging.getLogger(__name__)


class StoryAgent(BaseAgent):
    """故事智能体：负责保持故事性的压缩"""

    def __init__(self, llm_service=None, **kwargs):
        super().__init__(
            name="StoryAgent",
            timeout_seconds=120,
            **kwargs
        )
        self.llm_service = llm_service

    def get_task_types(self) -> List[str]:
        """获取支持的任务类型"""
        return ["story_compression", "narrative_analysis", "flow_optimization", "health_check"]

    async def process_task(self, task: AgentTask) -> AgentResponse:
        """处理故事任务"""
        try:
            if task.task_type == "story_compression":
                return await self._story_compression(task)
            elif task.task_type == "narrative_analysis":
                return await self._analyze_narrative(task)
            elif task.task_type == "flow_optimization":
                return await self._optimize_flow(task)
            elif task.task_type == "health_check":
                return AgentResponse(
                    success=True,
                    result={"status": "healthy", "service": "story"},
                    agent_name=self.name,
                    task_type=task.task_type
                )
            else:
                raise AgentError(f"不支持的任务类型: {task.task_type}")

        except Exception as e:
            logger.error(f"故事任务处理失败: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name,
                task_type=task.task_type
            )

    async def _story_compression(self, task: AgentTask) -> AgentResponse:
        """执行故事压缩"""
        # 模拟故事压缩结果
        result = StoryCompressionResult(
            condensed_narratives={"开场": "精简的开场叙述"},
            streamlined_plot="优化后的情节线",
            emotional_impact_score=0.8,
            story_coherence_score=0.9,
            compression_ratio=0.6,
            artistic_reasoning="保持情感冲击力，精简冗余叙述"
        )

        return AgentResponse(
            success=True,
            result=result.dict(),
            agent_name=self.name,
            task_type=task.task_type
        )

    async def _analyze_narrative(self, task: AgentTask) -> AgentResponse:
        """分析叙述结构"""
        # 模拟叙述分析
        narrative_structure = {
            "acts": 3,
            "turning_points": 2,
            "climax_position": 0.8,
            "resolution_type": "closed"
        }

        return AgentResponse(
            success=True,
            result={"narrative_structure": narrative_structure},
            agent_name=self.name,
            task_type=task.task_type
        )

    async def _optimize_flow(self, task: AgentTask) -> AgentResponse:
        """优化故事流程"""
        # 模拟流程优化
        optimizations = [
            "重新排列事件顺序",
            "合并相似场景",
            "强化节奏感"
        ]

        return AgentResponse(
            success=True,
            result={"optimizations": optimizations},
            agent_name=self.name,
            task_type=task.task_type
        )