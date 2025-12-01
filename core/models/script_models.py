"""剧本相关数据模型"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class EntityType(str, Enum):
    """实体类型枚举"""
    PERSON = "Person"
    LOCATION = "Location"
    ITEM = "Item"
    EVENT = "Event"
    TIMELINE = "Timeline"


class RelationType(str, Enum):
    """关系类型枚举"""
    LOCATED_IN = "LOCATED_IN"
    HAS_ITEM = "HAS_ITEM"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    HAPPENED_AT = "HAPPENED_AT"
    HAPPENED_ON = "HAPPENED_ON"
    KNOWS = "KNOWS"
    KILLED = "KILLED"
    SUSPECTS = "SUSPECTS"


class ScriptEntity(BaseModel):
    """剧本实体模型"""
    id: str = Field(..., description="实体唯一标识")
    type: EntityType = Field(..., description="实体类型")
    name: str = Field(..., description="实体名称")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    description: Optional[str] = Field(None, description="实体描述")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="提取置信度")


class ScriptRelation(BaseModel):
    """剧本关系模型"""
    id: str = Field(..., description="关系唯一标识")
    source_entity_id: str = Field(..., description="源实体ID")
    target_entity_id: str = Field(..., description="目标实体ID")
    relation_type: RelationType = Field(..., description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")
    description: Optional[str] = Field(None, description="关系描述")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="提取置信度")


class ScriptEvent(BaseModel):
    """剧本事件模型"""
    id: str = Field(..., description="事件唯一标识")
    name: str = Field(..., description="事件名称")
    description: str = Field(..., description="事件描述")
    location_id: Optional[str] = Field(None, description="发生地点ID")
    participants: List[str] = Field(default_factory=list, description="参与者ID列表")
    timeline: Optional[str] = Field(None, description="时间线")
    importance: float = Field(default=1.0, ge=0.0, le=1.0, description="事件重要性")
    is_critical: bool = Field(default=False, description="是否关键事件")


class Timeline(BaseModel):
    """时间线模型"""
    id: str = Field(..., description="时间线ID")
    name: str = Field(..., description="时间线名称")
    start_time: str = Field(..., description="开始时间")
    end_time: str = Field(..., description="结束时间")
    events: List[ScriptEvent] = Field(default_factory=list, description="事件列表")
    description: Optional[str] = Field(None, description="时间线描述")


class ScriptMetadata(BaseModel):
    """剧本元数据"""
    title: str = Field(..., description="剧本标题")
    author: Optional[str] = Field(None, description="作者")
    genre: Optional[str] = Field(None, description="类型")
    difficulty: Optional[str] = Field(None, description="难度")
    player_count_min: int = Field(default=1, ge=1, description="最少玩家数")
    player_count_max: int = Field(default=1, ge=1, description="最多玩家数")
    estimated_duration_hours: float = Field(..., description="预估时长（小时）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class PlayerScript(BaseModel):
    """玩家剧本模型"""
    player_id: str = Field(..., description="玩家ID")
    player_name: str = Field(..., description="玩家角色名")
    content: str = Field(..., description="剧本内容")
    background: Optional[str] = Field(None, description="角色背景")
    objectives: List[str] = Field(default_factory=list, description="任务目标")
    clues: List[str] = Field(default_factory=list, description="线索信息")
    secrets: List[str] = Field(default_factory=list, description="秘密信息")
    relationships: Dict[str, str] = Field(default_factory=dict, description="人物关系")


class MasterScript(BaseModel):
    """主持人剧本模型"""
    content: str = Field(..., description="主持人剧本内容")
    timeline: str = Field(..., description="完整时间线")
    truth: str = Field(..., description="案件真相")
    key_clues: List[str] = Field(default_factory=list, description="关键线索")
    red_herrings: List[str] = Field(default_factory=list, description="干扰信息")
    solution_steps: List[str] = Field(default_factory=list, description="解答步骤")


class Script(BaseModel):
    """完整剧本模型"""
    id: str = Field(..., description="剧本唯一标识")
    metadata: ScriptMetadata = Field(..., description="剧本元数据")
    player_scripts: Dict[str, PlayerScript] = Field(..., description="玩家剧本字典")
    master_script: MasterScript = Field(..., description="主持人剧本")
    entities: List[ScriptEntity] = Field(default_factory=list, description="实体列表")
    relations: List[ScriptRelation] = Field(default_factory=list, description="关系列表")
    events: List[ScriptEvent] = Field(default_factory=list, description="事件列表")
    timelines: List[Timeline] = Field(default_factory=list, description="时间线列表")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ScriptExtractionResult(BaseModel):
    """剧本提取结果"""
    success: bool = Field(..., description="提取是否成功")
    script_id: str = Field(..., description="剧本ID")
    entities_count: int = Field(default=0, description="提取的实体数量")
    relations_count: int = Field(default=0, description="提取的关系数量")
    events_count: int = Field(default=0, description="提取的事件数量")
    processing_time: float = Field(..., description="处理时间（秒）")
    errors: List[str] = Field(default_factory=list, description="错误信息")
    warnings: List[str] = Field(default_factory=list, description="警告信息")


class RAGQuery(BaseModel):
    """RAG查询模型"""
    question: str = Field(..., description="查询问题")
    script_id: Optional[str] = Field(None, description="剧本ID")
    max_context_length: int = Field(default=4000, description="最大上下文长度")
    search_k: int = Field(default=5, description="检索数量")
    include_graph: bool = Field(default=True, description="是否包含图谱检索")


class RAGResponse(BaseModel):
    """RAG响应模型"""
    answer: str = Field(..., description="生成的回答")
    sources: List[str] = Field(default_factory=list, description="引用的来源")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="回答置信度")
    query_time: float = Field(..., description="查询时间（秒）")
    graph_results: Optional[Dict[str, Any]] = Field(None, description="图谱检索结果")
    vector_results: Optional[List[Dict[str, Any]]] = Field(None, description="向量检索结果")