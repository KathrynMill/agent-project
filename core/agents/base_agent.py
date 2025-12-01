"""智能体基类模块"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio
import logging
from pydantic import BaseModel

from ..models.compression_models import AgentTask
from ...shared.utils.exceptions import AgentError, AgentTimeoutError

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """智能体响应模型"""
    success: bool = False
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    agent_name: str
    task_type: str
    metadata: Dict[str, Any] = {}


class BaseAgent(ABC):
    """智能体基类"""

    def __init__(
        self,
        name: str,
        timeout_seconds: int = 120,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化智能体

        Args:
            name: 智能体名称
            timeout_seconds: 超时时间（秒）
            retry_attempts: 重试次数
            retry_delay: 重试延迟（秒）
        """
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # 智能体状态
        self.is_busy = False
        self.current_task = None
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.average_processing_time = 0.0
        self.last_active_time = datetime.now()

        # 性能统计
        self.performance_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "error_types": {},
            "last_error": None
        }

    @abstractmethod
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """
        处理任务的抽象方法，子类必须实现

        Args:
            task: 要处理的任务

        Returns:
            AgentResponse: 处理结果
        """
        pass

    @abstractmethod
    def get_task_types(self) -> List[str]:
        """
        获取该智能体支持的任务类型

        Returns:
            List[str]: 支持的任务类型列表
        """
        pass

    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """
        执行任务的主要方法，包含重试和超时处理

        Args:
            task: 要执行的任务

        Returns:
            AgentResponse: 执行结果
        """
        if self.is_busy:
            raise AgentError(f"智能体 {self.name} 正忙，无法接受新任务")

        self.is_busy = True
        self.current_task = task
        start_time = datetime.now()

        try:
            self.logger.info(f"开始执行任务 {task.task_id}，类型: {task.task_type}")

            # 验证任务类型
            if task.task_type not in self.get_task_types():
                raise AgentError(f"智能体 {self.name} 不支持任务类型: {task.task_type}")

            # 执行任务，带重试机制
            response = await self._execute_with_retry(task)

            # 更新性能统计
            self._update_performance_stats(response, start_time)

            self.logger.info(f"任务 {task.task_id} 执行完成，成功: {response.success}")
            return response

        except Exception as e:
            self.logger.error(f"任务 {task.task_id} 执行失败: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds(),
                agent_name=self.name,
                task_type=task.task_type,
                metadata={"error_type": type(e).__name__}
            )
        finally:
            self.is_busy = False
            self.current_task = None
            self.last_active_time = datetime.now()

    async def _execute_with_retry(self, task: AgentTask) -> AgentResponse:
        """带重试机制的任务执行"""
        last_error = None

        for attempt in range(self.retry_attempts + 1):
            try:
                # 使用超时执行
                return await asyncio.wait_for(
                    self.process_task(task),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                error_msg = f"任务 {task.task_id} 执行超时（{self.timeout_seconds}秒）"
                self.logger.warning(f"第 {attempt + 1} 次尝试超时: {error_msg}")
                last_error = AgentTimeoutError(error_msg)
            except Exception as e:
                error_msg = f"任务 {task.task_id} 执行失败: {str(e)}"
                self.logger.warning(f"第 {attempt + 1} 次尝试失败: {error_msg}")
                last_error = e

            # 如果不是最后一次尝试，等待后重试
            if attempt < self.retry_attempts:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        # 所有重试都失败了
        if isinstance(last_error, asyncio.TimeoutError):
            raise AgentTimeoutError(
                f"任务 {task.task_id} 在 {self.retry_attempts + 1} 次尝试后仍然超时",
                "AGENT_TIMEOUT"
            )
        else:
            raise AgentError(
                f"任务 {task.task_id} 在 {self.retry_attempts + 1} 次尝试后仍然失败: {str(last_error)}",
                "AGENT_EXECUTION_FAILED"
            )

    def _update_performance_stats(self, response: AgentResponse, start_time: datetime):
        """更新性能统计信息"""
        processing_time = (datetime.now() - start_time).total_seconds()

        self.performance_stats["total_tasks"] += 1
        self.performance_stats["total_processing_time"] += processing_time

        if response.success:
            self.performance_stats["successful_tasks"] += 1
        else:
            self.performance_stats["failed_tasks"] += 1
            # 记录错误类型
            error_type = response.metadata.get("error_type", "Unknown")
            self.performance_stats["error_types"][error_type] = \
                self.performance_stats["error_types"].get(error_type, 0) + 1
            self.performance_stats["last_error"] = response.error

        # 计算平均处理时间
        if self.performance_stats["total_tasks"] > 0:
            self.performance_stats["average_processing_time"] = \
                self.performance_stats["total_processing_time"] / self.performance_stats["total_tasks"]

    def get_status(self) -> Dict[str, Any]:
        """
        获取智能体状态信息

        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            "name": self.name,
            "is_busy": self.is_busy,
            "current_task_id": self.current_task.task_id if self.current_task else None,
            "last_active_time": self.last_active_time.isoformat(),
            "performance_stats": self.performance_stats.copy()
        }

    def get_supported_task_types(self) -> List[str]:
        """获取支持的任务类型（代理方法）"""
        return self.get_task_types()

    def can_handle_task(self, task_type: str) -> bool:
        """
        检查是否能处理指定类型的任务

        Args:
            task_type: 任务类型

        Returns:
            bool: 是否能处理
        """
        return task_type in self.get_task_types()

    def reset_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "error_types": {},
            "last_error": None
        }
        self.logger.info(f"智能体 {self.name} 性能统计已重置")

    def validate_task(self, task: AgentTask) -> bool:
        """
        验证任务的有效性

        Args:
            task: 要验证的任务

        Returns:
            bool: 任务是否有效
        """
        # 基本验证
        if not task.task_id or not task.task_type or not task.input_data:
            return False

        # 检查是否支持该任务类型
        if task.task_type not in self.get_task_types():
            return False

        # 子类可以重写此方法进行更详细的验证
        return True

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            # 尝试执行一个简单的测试任务
            test_task = AgentTask(
                task_id="health_check",
                agent_name=self.name,
                task_type="health_check",
                input_data={"test": True},
                timeout_seconds=5
            )

            if self.get_task_types():
                # 如果有可用的任务类型，尝试处理健康检查
                await asyncio.wait_for(self.process_task(test_task), timeout=5)

            return {
                "status": "healthy",
                "agent_name": self.name,
                "can_accept_tasks": not self.is_busy,
                "last_active": self.last_active_time.isoformat(),
                "supported_tasks": self.get_task_types()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent_name": self.name,
                "error": str(e),
                "last_active": self.last_active_time.isoformat()
            }

    def __str__(self) -> str:
        return f"Agent({self.name})"

    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', busy={self.is_busy}, tasks={self.performance_stats['total_tasks']})"