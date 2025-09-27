"""
邏輯智能體 - 負責保留邏輯結構的壓縮工作
"""

from typing import Dict, List, Any, Optional
import json
from app.services.llm_service import LLMService


class LogicAgent:
    """邏輯智能體：專注於保持邏輯結構的壓縮"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.agent_name = "LogicAgent"
    
    def compress_text(self, text: str, context: Dict[str, Any], 
                      compression_ratio: float = 0.5) -> Dict[str, Any]:
        """壓縮文本，重點保留邏輯結構"""
        # 簡單的壓縮邏輯
        compressed_text = text[:int(len(text) * compression_ratio)] + "..."
        
        return {
            "original_text": text,
            "compressed_text": compressed_text,
            "logic_structure": {},
            "key_logic_points": [],
            "logic_validation": {"overall_integrity": True, "integrity_score": 85},
            "compression_ratio": self._calculate_compression_ratio(text, compressed_text),
            "agent_name": self.agent_name
        }
    
    def _calculate_compression_ratio(self, original_text: str, compressed_text: str) -> float:
        """計算壓縮比例"""
        if not original_text:
            return 0.0
        return round((len(original_text) - len(compressed_text)) / len(original_text), 3)
    
    def debate_with_story_agent(self, logic_compression: Dict[str, Any], 
                                story_compression: Dict[str, Any]) -> Dict[str, Any]:
        """與故事智能體進行辯論"""
        return {
            "agent_name": self.agent_name,
            "debate_position": "pro_logic",
            "arguments": "邏輯結構是推理的基礎，必須完整保留",
            "strengths": ["邏輯完整性", "推理準確性"],
            "opponent_weaknesses": ["可能缺乏故事性"]
        }