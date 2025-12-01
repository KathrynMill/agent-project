"""主编智能体 - 协调多智能体工作流程和压缩流程控制"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import logging
from enum import Enum

from .base_agent import BaseAgent, AgentResponse
from ..models.compression_models import (
    AgentTask, CompressionRequest, CompressionResult, CompressionProgress,
    CompressionState, AnalysisResult, LogicCompressionResult, StoryCompressionResult,
    DebateResult, ValidationResult
)
from ...shared.utils.exceptions import AgentError, WorkflowError, CompressionError
from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class WorkflowStep(str, Enum):
    """工作流程步骤枚举"""
    ANALYZE = "analyze"
    PLAN = "plan"
    LOGIC_COMPRESS = "logic_compress"
    STORY_COMPRESS = "story_compress"
    DEBATE = "debate"
    INTEGRATE = "integrate"
    VALIDATE = "validate"
    FINALIZE = "finalize"
    FAILED = "failed"


class ChiefEditorAgent(BaseAgent):
    """主编智能体：协调整个压缩工作流程"""

    def __init__(
        self,
        analysis_agent: 'AnalysisAgent',
        logic_agent: 'LogicAgent',
        story_agent: 'StoryAgent',
        validation_agent: 'ValidationAgent',
        **kwargs
    ):
        """
        初始化主编智能体

        Args:
            analysis_agent: 分析智能体
            logic_agent: 逻辑智能体
            story_agent: 故事智能体
            validation_agent: 验证智能体
        """
        super().__init__(
            name="ChiefEditorAgent",
            timeout_seconds=settings.compression.agent_timeout * 3,  # 主编需要更长时间
            retry_attempts=2,
            **kwargs
        )

        # 注册子智能体
        self.agents = {
            "analysis": analysis_agent,
            "logic": logic_agent,
            "story": story_agent,
            "validation": validation_agent
        }

        # 工作流程状态跟踪
        self.active_workflows: Dict[str, CompressionState] = {}
        self.workflow_history: List[Dict[str, Any]] = []

        self.logger.info("主编智能体初始化完成")

    def get_task_types(self) -> List[str]:
        """获取支持的任务类型"""
        return [
            "compress_script",
            "workflow_control",
            "coordinate_agents",
            "integration",
            "health_check"
        ]

    async def process_task(self, task: AgentTask) -> AgentResponse:
        """处理主编任务"""
        try:
            if task.task_type == "compress_script":
                return await self._handle_compression_task(task)
            elif task.task_type == "workflow_control":
                return await self._handle_workflow_control(task)
            elif task.task_type == "coordinate_agents":
                return await self._handle_agent_coordination(task)
            elif task.task_type == "integration":
                return await self._handle_integration_task(task)
            elif task.task_type == "health_check":
                return AgentResponse(
                    success=True,
                    result={"status": "healthy", "agents": len(self.agents)},
                    agent_name=self.name,
                    task_type=task.task_type
                )
            else:
                raise AgentError(f"不支持的任务类型: {task.task_type}")

        except Exception as e:
            self.logger.error(f"主编任务处理失败: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name,
                task_type=task.task_type
            )

    async def _handle_compression_task(self, task: AgentTask) -> AgentResponse:
        """处理剧本压缩任务"""
        start_time = datetime.now()
        compression_request = CompressionRequest(**task.input_data)

        try:
            # 创建压缩状态
            state = self._create_compression_state(compression_request)
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.active_workflows[workflow_id] = state

            # 执行压缩工作流程
            result = await self._execute_workflow(state, workflow_id)

            # 记录工作流程历史
            workflow_record = {
                "workflow_id": workflow_id,
                "start_time": start_time,
                "end_time": datetime.now(),
                "success": result.success,
                "duration": (datetime.now() - start_time).total_seconds(),
                "iterations": result.iterations if result.success else 0
            }
            self.workflow_history.append(workflow_record)

            return AgentResponse(
                success=result.success,
                result=result.dict() if result.success else {"error": str(result.error) if hasattr(result, 'error') else "压缩失败"},
                processing_time=(datetime.now() - start_time).total_seconds(),
                agent_name=self.name,
                task_type=task.task_type,
                metadata={"workflow_id": workflow_id}
            )

        except Exception as e:
            self.logger.error(f"压缩任务执行失败: {str(e)}")
            return AgentResponse(
                success=False,
                error=f"压缩任务执行失败: {str(e)}",
                processing_time=(datetime.now() - start_time).total_seconds(),
                agent_name=self.name,
                task_type=task.task_type
            )
        finally:
            # 清理活跃的工作流程
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

    def _create_compression_state(self, request: CompressionRequest) -> CompressionState:
        """创建压缩工作流程状态"""
        return CompressionState(
            script=request.script,
            target_hours=request.target_hours,
            compression_level=request.compression_level,
            strategy=request.strategy,
            preserve_elements=request.preserve_elements,
            remove_elements=request.remove_elements,
            custom_requirements=request.custom_requirements,
            current_step=WorkflowStep.ANALYZE,
            iteration_count=0,
            max_iterations=settings.compression.max_iterations,
            error_messages=[],
            start_time=datetime.now()
        )

    async def _execute_workflow(self, state: CompressionState, workflow_id: str) -> CompressionResult:
        """执行完整的压缩工作流程"""
        self.logger.info(f"开始执行工作流程 {workflow_id}")

        try:
            # 步骤1: 分析剧本
            state.current_step = WorkflowStep.ANALYZE
            analysis_result = await self._execute_analysis_step(state)
            state.analysis_result = analysis_result

            # 步骤2: 制定压缩策略
            state.current_step = WorkflowStep.PLAN
            compression_strategy = await self._execute_planning_step(state, analysis_result)
            state.compression_strategy = compression_strategy

            # 迭代执行压缩和验证
            for iteration in range(settings.compression.max_iterations):
                state.iteration_count = iteration + 1
                self.logger.info(f"开始第 {iteration + 1} 次压缩迭代")

                # 并行执行逻辑压缩和故事压缩
                state.current_step = WorkflowStep.LOGIC_COMPRESS
                logic_task = self._execute_logic_compression_step(state)

                state.current_step = WorkflowStep.STORY_COMPRESS
                story_task = self._execute_story_compression_step(state)

                # 等待两个压缩任务完成
                logic_result, story_result = await asyncio.gather(
                    logic_task, story_task, return_exceptions=True
                )

                # 检查压缩结果
                if isinstance(logic_result, Exception):
                    raise CompressionError(f"逻辑压缩失败: {str(logic_result)}")
                if isinstance(story_result, Exception):
                    raise CompressionError(f"故事压缩失败: {str(story_result)}")

                state.logic_compression = logic_result
                state.story_compression = story_result

                # 步骤5: 智能体辩论
                state.current_step = WorkflowStep.DEBATE
                debate_result = await self._execute_debate_step(state, logic_result, story_result)
                state.debate_result = debate_result

                # 步骤6: 整合结果
                state.current_step = WorkflowStep.INTEGRATE
                integrated_result = await self._execute_integration_step(state, debate_result)
                state.final_compression = integrated_result

                # 步骤7: 验证结果
                state.current_step = WorkflowStep.VALIDATE
                validation_result = await self._execute_validation_step(state, integrated_result)
                state.validation_result = validation_result

                # 检查是否需要继续迭代
                if validation_result.is_acceptable:
                    break
                elif iteration == settings.compression.max_iterations - 1:
                    self.logger.warning(f"达到最大迭代次数 {settings.compression.max_iterations}，终止流程")
                    break
                else:
                    self.logger.info(f"验证未通过，开始第 {iteration + 2} 次迭代")

            # 最终步骤
            state.current_step = WorkflowStep.FINALIZE
            final_result = await self._execute_finalization_step(state)

            self.logger.info(f"工作流程 {workflow_id} 执行完成")
            return final_result

        except Exception as e:
            self.logger.error(f"工作流程 {workflow_id} 执行失败: {str(e)}")
            state.current_step = WorkflowStep.FAILED
            state.error_messages.append(str(e))
            raise CompressionError(f"工作流程执行失败: {str(e)}")

    async def _execute_analysis_step(self, state: CompressionState) -> AnalysisResult:
        """执行分析步骤"""
        task = AgentTask(
            task_id=f"analysis_{state.script.id}_{datetime.now().strftime('%H%M%S')}",
            agent_name="AnalysisAgent",
            task_type="analyze_script",
            input_data={
                "script": state.script.dict(),
                "target_hours": state.target_hours
            }
        )

        response = await self.agents["analysis"].execute_task(task)

        if not response.success:
            raise WorkflowError(f"剧本分析失败: {response.error}")

        return AnalysisResult(**response.result)

    async def _execute_planning_step(self, state: CompressionState, analysis: AnalysisResult) -> Dict[str, Any]:
        """执行策略制定步骤"""
        # 根据分析结果制定压缩策略
        strategy = {
            "strategy": analysis.recommended_strategy,
            "target_ratio": state.target_hours / state.script.metadata.estimated_duration_hours,
            "key_elements": state.preserve_elements,
            "removable_elements": state.remove_elements,
            "priority_areas": analysis.key_entities,
            "compression_focus": self._determine_compression_focus(analysis)
        }

        return strategy

    def _determine_compression_focus(self, analysis: AnalysisResult) -> List[str]:
        """根据分析结果确定压缩重点"""
        focus = []

        if analysis.script_complexity > 0.7:
            focus.append("simplify_relationships")

        if len(analysis.critical_events) > len(analysis.story_beats):
            focus.append("reduce_non_critical_events")

        focus.append("condense_narratives")
        focus.append("merge_similar_entities")

        return focus

    async def _execute_logic_compression_step(self, state: CompressionState) -> LogicCompressionResult:
        """执行逻辑压缩步骤"""
        task = AgentTask(
            task_id=f"logic_compress_{state.script.id}_{datetime.now().strftime('%H%M%S')}",
            agent_name="LogicAgent",
            task_type="logic_compression",
            input_data={
                "script": state.script.dict(),
                "strategy": state.compression_strategy,
                "analysis": state.analysis_result.dict() if state.analysis_result else {}
            }
        )

        response = await self.agents["logic"].execute_task(task)

        if not response.success:
            raise WorkflowError(f"逻辑压缩失败: {response.error}")

        return LogicCompressionResult(**response.result)

    async def _execute_story_compression_step(self, state: CompressionState) -> StoryCompressionResult:
        """执行故事压缩步骤"""
        task = AgentTask(
            task_id=f"story_compress_{state.script.id}_{datetime.now().strftime('%H%M%S')}",
            agent_name="StoryAgent",
            task_type="story_compression",
            input_data={
                "script": state.script.dict(),
                "strategy": state.compression_strategy,
                "analysis": state.analysis_result.dict() if state.analysis_result else {}
            }
        )

        response = await self.agents["story"].execute_task(task)

        if not response.success:
            raise WorkflowError(f"故事压缩失败: {response.error}")

        return StoryCompressionResult(**response.result)

    async def _execute_debate_step(
        self,
        state: CompressionState,
        logic_result: LogicCompressionResult,
        story_result: StoryCompressionResult
    ) -> DebateResult:
        """执行智能体辩论步骤"""
        # 这里可以实现智能体之间的辩论逻辑
        # 目前简化为基于评分的决策

        logic_score = logic_result.logic_score
        story_score = story_result.story_coherence_score

        if logic_score > story_score:
            winner = "logic_compression"
            winner_score = logic_score
        else:
            winner = "story_compression"
            winner_score = story_score

        consensus_score = min(logic_score, story_score)

        debate_result = DebateResult(
            winner=winner,
            consensus_score=consensus_score,
            debate_summary=f"逻辑压缩评分: {logic_score:.2f}, 故事压缩评分: {story_score:.2f}",
            final_strategy={
                "primary_approach": winner,
                "secondary_elements": "logic_compression" if winner == "story_compression" else "story_compression",
                "integration_method": "weighted_blend"
            },
            conflicts_resolved=[],
            remaining_conflicts=[]
        )

        return debate_result

    async def _execute_integration_step(
        self,
        state: CompressionState,
        debate_result: DebateResult
    ) -> Dict[str, Any]:
        """执行结果整合步骤"""
        # 根据辩论结果整合逻辑压缩和故事压缩的结果
        integrated_script = state.script.dict()  # 这里需要实际的整合逻辑

        return {
            "integrated_script": integrated_script,
            "integration_method": debate_result.final_strategy["integration_method"],
            "primary_approach": debate_result.winner,
            "quality_score": debate_result.consensus_score
        }

    async def _execute_validation_step(
        self,
        state: CompressionState,
        integrated_result: Dict[str, Any]
    ) -> ValidationResult:
        """执行验证步骤"""
        task = AgentTask(
            task_id=f"validation_{state.script.id}_{datetime.now().strftime('%H%M%S')}",
            agent_name="ValidationAgent",
            task_type="validate_compression",
            input_data={
                "original_script": state.script.dict(),
                "compressed_script": integrated_result["integrated_script"],
                "target_hours": state.target_hours
            }
        )

        response = await self.agents["validation"].execute_task(task)

        if not response.success:
            raise WorkflowError(f"压缩验证失败: {response.error}")

        return ValidationResult(**response.result)

    async def _execute_finalization_step(self, state: CompressionState) -> CompressionResult:
        """执行最终化步骤"""
        processing_time = (datetime.now() - state.start_time).total_seconds()

        # 构建最终结果
        result = CompressionResult(
            success=state.validation_result.is_acceptable if state.validation_result else False,
            compression_id=f"compression_{state.script.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            original_duration_hours=state.script.metadata.estimated_duration_hours,
            compressed_duration_hours=state.target_hours,
            compression_ratio=state.target_hours / state.script.metadata.estimated_duration_hours,

            # 这里需要从实际压缩结果中获取数据
            compressed_player_scripts={},
            compressed_master_script=state.script.master_script,

            analysis_result=state.analysis_result,
            logic_compression=state.logic_compression,
            story_compression=state.story_compression,
            debate_result=state.debate_result,
            validation_result=state.validation_result,

            processing_time=processing_time,
            iterations=state.iteration_count,
            quality_score=state.validation_result.overall_score if state.validation_result else 0.0,
            warnings=state.error_messages,
            recommendations=[],
            created_at=datetime.now()
        )

        return result

    async def _handle_workflow_control(self, task: AgentTask) -> AgentResponse:
        """处理工作流程控制任务"""
        control_type = task.input_data.get("control_type")

        if control_type == "get_status":
            return AgentResponse(
                success=True,
                result={
                    "active_workflows": len(self.active_workflows),
                    "workflow_history": len(self.workflow_history),
                    "agent_status": {name: agent.get_status() for name, agent in self.agents.items()}
                },
                agent_name=self.name,
                task_type=task.task_type
            )
        elif control_type == "cancel_workflow":
            workflow_id = task.input_data.get("workflow_id")
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
                return AgentResponse(
                    success=True,
                    result={"message": f"工作流程 {workflow_id} 已取消"},
                    agent_name=self.name,
                    task_type=task.task_type
                )
            else:
                return AgentResponse(
                    success=False,
                    error=f"工作流程 {workflow_id} 不存在",
                    agent_name=self.name,
                    task_type=task.task_type
                )
        else:
            return AgentResponse(
                success=False,
                error=f"不支持的控制类型: {control_type}",
                agent_name=self.name,
                task_type=task.task_type
            )

    async def _handle_agent_coordination(self, task: AgentTask) -> AgentResponse:
        """处理智能体协调任务"""
        coordination_type = task.input_data.get("coordination_type")

        if coordination_type == "parallel_execution":
            # 并行执行多个智能体任务
            tasks_data = task.input_data.get("tasks", [])
            tasks = []

            for task_data in tasks_data:
                agent_name = task_data.get("agent_name")
                if agent_name in self.agents:
                    agent_task = AgentTask(**task_data)
                    tasks.append(self.agents[agent_name].execute_task(agent_task))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return AgentResponse(
                    success=True,
                    result={"results": [r.dict() if hasattr(r, 'dict') else str(r) for r in results]},
                    agent_name=self.name,
                    task_type=task.task_type
                )
            else:
                return AgentResponse(
                    success=False,
                    error="没有有效的任务可执行",
                    agent_name=self.name,
                    task_type=task.task_type
                )
        else:
            return AgentResponse(
                success=False,
                error=f"不支持的协调类型: {coordination_type}",
                agent_name=self.name,
                task_type=task.task_type
            )

    async def _handle_integration_task(self, task: AgentTask) -> AgentResponse:
        """处理整合任务"""
        integration_type = task.input_data.get("integration_type")

        if integration_type == "merge_results":
            results = task.input_data.get("results", [])
            merge_strategy = task.input_data.get("merge_strategy", "weighted_average")

            # 这里实现具体的整合逻辑
            merged_result = {
                "merged_count": len(results),
                "strategy": merge_strategy,
                "timestamp": datetime.now().isoformat()
            }

            return AgentResponse(
                success=True,
                result=merged_result,
                agent_name=self.name,
                task_type=task.task_type
            )
        else:
            return AgentResponse(
                success=False,
                error=f"不支持的整合类型: {integration_type}",
                agent_name=self.name,
                task_type=task.task_type
            )

    def get_workflow_status(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """获取工作流程状态"""
        if workflow_id:
            if workflow_id in self.active_workflows:
                state = self.active_workflows[workflow_id]
                return {
                    "workflow_id": workflow_id,
                    "current_step": state.current_step,
                    "iteration_count": state.iteration_count,
                    "max_iterations": state.max_iterations,
                    "error_count": len(state.error_messages),
                    "start_time": state.start_time.isoformat(),
                    "status": "active"
                }
            else:
                return {"workflow_id": workflow_id, "status": "not_found"}
        else:
            return {
                "active_workflows": len(self.active_workflows),
                "workflow_ids": list(self.active_workflows.keys()),
                "total_completed": len(self.workflow_history),
                "agent_status": {name: not agent.is_busy for name, agent in self.agents.items()}
            }