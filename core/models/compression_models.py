"""压缩相关数据模型"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from .script_models import Script, PlayerScript, MasterScript


class CompressionLevel(str, Enum):
    """压缩级别枚举"""
    LIGHT = "light"      # 轻度压缩
    MEDIUM = "medium"    # 中度压缩
    HEAVY = "heavy"      # 重度压缩
    CUSTOM = "custom"    # 自定义压缩


class CompressionStrategy(str, Enum):
    """压缩策略枚举"""
    PRESERVE_LOGIC = "preserve_logic"    # 优先保持逻辑
    PRESERVE_STORY = "preserve_story"    # 优先保持故事
    BALANCED = "balanced"                # 平衡压缩
    FAST_COMPRESSION = "fast"            # 快速压缩


class AgentTask(BaseModel):
    """智能体任务模型"""
    task_id: str = Field(..., description="任务ID")
    agent_name: str = Field(..., description="智能体名称")
    task_type: str = Field(..., description="任务类型")
    input_data: Dict[str, Any] = Field(..., description="输入数据")
    priority: int = Field(default=1, ge=1, le=10, description="任务优先级")
    timeout_seconds: int = Field(default=120, description="超时时间（秒）")
    status: str = Field(default="pending", description="任务状态")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    error: Optional[str] = Field(None, description="错误信息")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")


class AnalysisResult(BaseModel):
    """分析结果模型"""
    script_complexity: float = Field(..., ge=0.0, le=1.0, description="剧本复杂度")
    key_entities: List[str] = Field(..., description="关键实体列表")
    critical_events: List[str] = Field(..., description="关键事件列表")
    logical_dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="逻辑依赖关系")
    story_beats: List[str] = Field(default_factory=list, description="故事节点")
    compression_potential: float = Field(..., ge=0.0, le=1.0, description="压缩潜力")
    analysis_summary: str = Field(..., description="分析摘要")
    recommended_strategy: CompressionStrategy = Field(..., description="推荐压缩策略")


class LogicCompressionResult(BaseModel):
    """逻辑压缩结果"""
    removed_events: List[str] = Field(default_factory=list, description="移除的事件列表")
    merged_entities: Dict[str, str] = Field(default_factory=dict, description="合并的实体映射")
    simplified_relations: List[str] = Field(default_factory=list, description="简化的关系列表")
    logic_score: float = Field(..., ge=0.0, le=1.0, description="逻辑一致性评分")
    compression_ratio: float = Field(..., ge=0.0, le=1.0, description="压缩比例")
    reasoning: str = Field(..., description="压缩理由")


class StoryCompressionResult(BaseModel):
    """故事压缩结果"""
    condensed_narratives: Dict[str, str] = Field(default_factory=dict, description="浓缩的叙述")
    streamlined_plot: str = Field(..., description="精简的情节")
    emotional_impact_score: float = Field(..., ge=0.0, le=1.0, description="情感冲击力评分")
    story_coherence_score: float = Field(..., ge=0.0, le=1.0, description="故事连贯性评分")
    compression_ratio: float = Field(..., ge=0.0, le=1.0, description="压缩比例")
    artistic_reasoning: str = Field(..., description="艺术处理理由")


class DebateResult(BaseModel):
    """智能体辩论结果"""
    winner: str = Field(..., description="胜出方案")
    consensus_score: float = Field(..., ge=0.0, le=1.0, description="共识度评分")
    debate_summary: str = Field(..., description="辩论摘要")
    final_strategy: Dict[str, Any] = Field(..., description="最终策略")
    conflicts_resolved: List[str] = Field(default_factory=list, description="已解决的冲突")
    remaining_conflicts: List[str] = Field(default_factory=list, description="剩余冲突")


class ValidationResult(BaseModel):
    """验证结果模型"""
    logic_consistency: bool = Field(..., description="逻辑一致性")
    story_completeness: bool = Field(..., description="故事完整性")
    player_experience: float = Field(..., ge=0.0, le=1.0, description="玩家体验评分")
    compression_quality: float = Field(..., ge=0.0, le=1.0, description="压缩质量评分")
    validation_issues: List[str] = Field(default_factory=list, description="验证发现的问题")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="综合评分")
    is_acceptable: bool = Field(..., description="是否可接受")


class CompressionRequest(BaseModel):
    """压缩请求模型"""
    script: Script = Field(..., description="原始剧本")
    target_hours: int = Field(..., ge=1, le=12, description="目标时长（小时）")
    compression_level: CompressionLevel = Field(default=CompressionLevel.MEDIUM, description="压缩级别")
    strategy: CompressionStrategy = Field(default=CompressionStrategy.BALANCED, description="压缩策略")
    preserve_elements: List[str] = Field(default_factory=list, description="必须保留的元素")
    remove_elements: List[str] = Field(default_factory=list, description="可以移除的元素")
    custom_requirements: Dict[str, Any] = Field(default_factory=dict, description="自定义要求")


class CompressionProgress(BaseModel):
    """压缩进度模型"""
    progress_id: str = Field(..., description="进度ID")
    current_step: str = Field(..., description="当前步骤")
    total_steps: int = Field(..., description="总步骤数")
    completed_steps: int = Field(..., description="已完成步骤数")
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="进度百分比")
    current_agent: Optional[str] = Field(None, description="当前工作的智能体")
    estimated_remaining_time: Optional[int] = Field(None, description="预估剩余时间（秒）")
    status_message: str = Field(..., description="状态消息")
    start_time: datetime = Field(default_factory=datetime.now, description="开始时间")
    last_update: datetime = Field(default_factory=datetime.now, description="最后更新时间")


class CompressionResult(BaseModel):
    """压缩结果模型"""
    success: bool = Field(..., description="压缩是否成功")
    compression_id: str = Field(..., description="压缩任务ID")
    original_duration_hours: float = Field(..., description="原始时长")
    compressed_duration_hours: float = Field(..., description="压缩后时长")
    compression_ratio: float = Field(..., description="压缩比例")

    # 压缩后的剧本
    compressed_player_scripts: Dict[str, PlayerScript] = Field(..., description="压缩后的玩家剧本")
    compressed_master_script: MasterScript = Field(..., description="压缩后的主持人剧本")

    # 各阶段结果
    analysis_result: Optional[AnalysisResult] = Field(None, description="分析结果")
    logic_compression: Optional[LogicCompressionResult] = Field(None, description="逻辑压缩结果")
    story_compression: Optional[StoryCompressionResult] = Field(None, description="故事压缩结果")
    debate_result: Optional[DebateResult] = Field(None, description="辩论结果")
    validation_result: Optional[ValidationResult] = Field(None, description="验证结果")

    # 元数据
    processing_time: float = Field(..., description="总处理时间（秒）")
    iterations: int = Field(default=1, description="迭代次数")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="压缩质量评分")
    warnings: List[str] = Field(default_factory=list, description="警告信息")
    recommendations: List[str] = Field(default_factory=list, description="优化建议")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class CompressionStatistics(BaseModel):
    """压缩统计模型"""
    total_compressions: int = Field(default=0, description="总压缩次数")
    successful_compressions: int = Field(default=0, description="成功压缩次数")
    average_compression_ratio: float = Field(default=0.0, description="平均压缩比例")
    average_quality_score: float = Field(default=0.0, description="平均质量评分")
    popular_compression_levels: Dict[str, int] = Field(default_factory=dict, description="各级别使用次数")
    common_issues: List[str] = Field(default_factory=list, description="常见问题")
    processing_time_stats: Dict[str, float] = Field(default_factory=dict, description="处理时间统计")


class AgentPerformance(BaseModel):
    """智能体性能模型"""
    agent_name: str = Field(..., description="智能体名称")
    total_tasks: int = Field(default=0, description="总任务数")
    successful_tasks: int = Field(default=0, description="成功任务数")
    average_task_time: float = Field(default=0.0, description="平均任务时间")
    quality_score: float = Field(default=0.0, description="工作质量评分")
    common_errors: List[str] = Field(default_factory=list, description="常见错误")
    last_active: Optional[datetime] = Field(None, description="最后活跃时间")


class SystemHealth(BaseModel):
    """系统健康状态模型"""
    overall_status: str = Field(..., description="总体状态")
    cpu_usage: float = Field(..., ge=0.0, le=100.0, description="CPU使用率")
    memory_usage: float = Field(..., ge=0.0, le=100.0, description="内存使用率")
    active_compressions: int = Field(default=0, description="活跃压缩任务数")
    queued_tasks: int = Field(default=0, description="排队任务数")
    agent_status: Dict[str, str] = Field(default_factory=dict, description="智能体状态")
    service_status: Dict[str, str] = Field(default_factory=dict, description="服务状态")
    last_check: datetime = Field(default_factory=datetime.now, description="最后检查时间")