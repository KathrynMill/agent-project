"""逻辑智能体 - 负责保持逻辑一致性的压缩"""

from typing import Dict, List, Any
import logging

from .base_agent import BaseAgent, AgentResponse
from ..models.compression_models import AgentTask, LogicCompressionResult

logger = logging.getLogger(__name__)


class LogicAgent(BaseAgent):
    """逻辑智能体：负责保持逻辑一致性的压缩"""

    def __init__(self, llm_service=None, **kwargs):
        super().__init__(
            name="LogicAgent",
            timeout_seconds=120,
            **kwargs
        )
        self.llm_service = llm_service

    def get_task_types(self) -> List[str]:
        """获取支持的任务类型"""
        return ["logic_compression", "consistency_check", "dependency_analysis", "health_check"]

    async def process_task(self, task: AgentTask) -> AgentResponse:
        """处理逻辑任务"""
        try:
            if task.task_type == "logic_compression":
                return await self._logic_compression(task)
            elif task.task_type == "consistency_check":
                return await self._check_consistency(task)
            elif task.task_type == "dependency_analysis":
                return await self._analyze_dependencies(task)
            elif task.task_type == "health_check":
                return AgentResponse(
                    success=True,
                    result={"status": "healthy", "service": "logic"},
                    agent_name=self.name,
                    task_type=task.task_type
                )
            else:
                raise AgentError(f"不支持的任务类型: {task.task_type}")

        except Exception as e:
            logger.error(f"逻辑任务处理失败: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name,
                task_type=task.task_type
            )

    async def _logic_compression(self, task: AgentTask) -> AgentResponse:
        """执行逻辑压缩"""
        # 模拟逻辑压缩结果
        result = LogicCompressionResult(
            removed_events=["次要对话", "重复描述"],
            merged_entities={"A": "B"},  # 实体A合并到实体B
            simplified_relations=["移除次要关系"],
            logic_score=0.85,
            compression_ratio=0.7,
            reasoning="保持核心逻辑链条，移除冗余信息"
        )

        return AgentResponse(
            success=True,
            result=result.dict(),
            agent_name=self.name,
            task_type=task.task_type
        )

    async def _check_consistency(self, task: AgentTask) -> AgentResponse:
        """检查逻辑一致性"""
        # 模拟一致性检查
        is_consistent = True
        issues = []

        return AgentResponse(
            success=True,
            result={"consistent": is_consistent, "issues": issues},
            agent_name=self.name,
            task_type=task.task_type
        )

    async def _analyze_dependencies(self, task: AgentTask) -> AgentResponse:
        """分析依赖关系"""
        # 模拟依赖分析
        dependencies = {
            "事件A": ["事件B", "事件C"],
            "人物关系": ["时间线"]
        }

        return AgentResponse(
            success=True,
            result={"dependencies": dependencies},
            agent_name=self.name,
            task_type=task.task_type
        )