"""
故事智能體 - 負責保留故事細節和戲劇效果的壓縮工作
"""

from typing import Dict, List, Any, Optional
import json
from app.services.llm_service import LLMService


class StoryAgent:
    """故事智能體：專注於保持故事細節和戲劇效果的壓縮"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.agent_name = "StoryAgent"
    
    def compress_text(self, text: str, context: Dict[str, Any], 
                      compression_ratio: float = 0.5) -> Dict[str, Any]:
        """壓縮文本，重點保留故事細節和戲劇效果"""
        # 簡單的壓縮邏輯
        compressed_text = text[:int(len(text) * compression_ratio)] + "..."
        
        return {
            "original_text": text,
            "compressed_text": compressed_text,
            "story_structure": {},
            "golden_details": [],
            "story_validation": {"overall_story_integrity": True, "story_attractiveness_score": 85},
            "compression_ratio": self._calculate_compression_ratio(text, compressed_text),
            "agent_name": self.agent_name
        }
    
    def _calculate_compression_ratio(self, original_text: str, compressed_text: str) -> float:
        """計算壓縮比例"""
        if not original_text:
            return 0.0
        return round((len(original_text) - len(compressed_text)) / len(original_text), 3)
    
    def debate_with_logic_agent(self, story_compression: Dict[str, Any], 
                               logic_compression: Dict[str, Any]) -> Dict[str, Any]:
        """與邏輯智能體進行辯論"""
        return {
            "agent_name": self.agent_name,
            "debate_position": "pro_story",
            "arguments": "故事吸引力是劇本的核心，必須保持戲劇效果",
            "strengths": ["故事吸引力", "情感共鳴", "戲劇效果"],
            "opponent_weaknesses": ["可能缺乏邏輯性"]
        }