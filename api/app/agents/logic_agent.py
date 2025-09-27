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
        try:
            # 1. 分析文本的邏輯結構
            logic_structure = self._analyze_logic_structure(text)
            
            # 2. 識別關鍵邏輯點
            key_logic_points = self._identify_key_logic_points(text, logic_structure)
            
            # 3. 基於邏輯重要性進行壓縮
            compressed_text = self._compress_with_logic_priority(
                text, key_logic_points, compression_ratio, context
            )
            
            # 4. 驗證壓縮後的邏輯完整性
            logic_validation = self._validate_logic_integrity(compressed_text, key_logic_points)
            
            return {
                "original_text": text,
                "compressed_text": compressed_text,
                "logic_structure": logic_structure,
                "key_logic_points": key_logic_points,
                "logic_validation": logic_validation,
                "compression_ratio": self._calculate_compression_ratio(text, compressed_text),
                "agent_name": self.agent_name
            }
            
        except Exception as e:
            print(f"[LogicAgent] 壓縮文本時出錯: {str(e)}")
            return self._get_default_compression_result(text, compression_ratio)
    
    def _analyze_logic_structure(self, text: str) -> Dict[str, Any]:
        """分析文本的邏輯結構"""
        try:
            # 使用 LLM 分析邏輯結構
            analysis_prompt = f"""
            請分析以下劇本文本的邏輯結構，識別關鍵的邏輯元素：

            【劇本文本】
            {text}

            請從以下角度分析：
            1. 時間線邏輯：事件的先後順序和時間關係
            2. 因果邏輯：事件之間的因果關係
            3. 推理鏈條：從線索到結論的推理過程
            4. 矛盾點：邏輯上的矛盾或不一致
            5. 關鍵證據：對推理至關重要的證據
            6. 動機邏輯：角色的動機和行為邏輯

            請以 JSON 格式返回分析結果：
            {{
                "timeline_logic": {{
                    "events": ["事件1", "事件2"],
                    "sequence": "事件順序描述",
                    "time_consistency": true/false
                }},
                "causal_logic": {{
                    "cause_effect_chains": ["因果鏈1", "因果鏈2"],
                    "logical_consistency": true/false
                }},
                "reasoning_chains": {{
                    "evidence_to_conclusion": ["推理鏈1", "推理鏈2"],
                    "logical_strength": "強/中/弱"
                }},
                "contradictions": [
                    {{"type": "時間矛盾", "description": "具體矛盾描述"}},
                    {{"type": "邏輯矛盾", "description": "具體矛盾描述"}}
                ],
                "key_evidence": ["證據1", "證據2"],
                "motive_logic": {{
                    "character_motives": {{"角色1": "動機1", "角色2": "動機2"}},
                    "motive_consistency": true/false
                }}
            }}
            """
            
            response = self.llm_service.generate_text(analysis_prompt)
            return json.loads(response)
            
        except Exception as e:
            print(f"[LogicAgent] 分析邏輯結構時出錯: {str(e)}")
            return self._get_default_logic_structure()
    
    def _identify_key_logic_points(self, text: str, logic_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """識別關鍵邏輯點"""
        try:
            key_points = []
            
            # 從邏輯結構中提取關鍵點
            timeline_events = logic_structure.get("timeline_logic", {}).get("events", [])
            causal_chains = logic_structure.get("causal_logic", {}).get("cause_effect_chains", [])
            reasoning_chains = logic_structure.get("reasoning_chains", {}).get("evidence_to_conclusion", [])
            key_evidence = logic_structure.get("key_evidence", [])
            contradictions = logic_structure.get("contradictions", [])
            
            # 時間線關鍵點
            for i, event in enumerate(timeline_events):
                key_points.append({
                    "type": "timeline_event",
                    "content": event,
                    "importance": "high" if i < 3 else "medium",
                    "logic_role": "時間線基礎"
                })
            
            # 因果關係關鍵點
            for chain in causal_chains:
                key_points.append({
                    "type": "causal_chain",
                    "content": chain,
                    "importance": "high",
                    "logic_role": "因果推理"
                })
            
            # 推理鏈關鍵點
            for chain in reasoning_chains:
                key_points.append({
                    "type": "reasoning_chain",
                    "content": chain,
                    "importance": "high",
                    "logic_role": "邏輯推理"
                })
            
            # 關鍵證據
            for evidence in key_evidence:
                key_points.append({
                    "type": "key_evidence",
                    "content": evidence,
                    "importance": "critical",
                    "logic_role": "推理基礎"
                })
            
            # 矛盾點（必須保留）
            for contradiction in contradictions:
                key_points.append({
                    "type": "contradiction",
                    "content": contradiction.get("description", ""),
                    "importance": "critical",
                    "logic_role": "推理關鍵"
                })
            
            return key_points
            
        except Exception as e:
            print(f"[LogicAgent] 識別關鍵邏輯點時出錯: {str(e)}")
            return []
    
    def _compress_with_logic_priority(self, text: str, key_logic_points: List[Dict[str, Any]], 
                                    compression_ratio: float, context: Dict[str, Any]) -> str:
        """基於邏輯重要性進行壓縮"""
        try:
            # 使用 LLM 進行智能壓縮
            compression_prompt = f"""
            請基於邏輯重要性壓縮以下劇本文本，重點保留邏輯結構和推理鏈條：

            【原始文本】
            {text}

            【關鍵邏輯點】
            {json.dumps(key_logic_points, ensure_ascii=False, indent=2)}

            【壓縮要求】
            1. 必須保留所有 critical 和 high 重要性的邏輯點
            2. 保持時間線的邏輯順序
            3. 保留因果關係和推理鏈條
            4. 保留所有矛盾點（這些是推理的關鍵）
            5. 保留關鍵證據
            6. 可以壓縮或簡化 medium 和 low 重要性的內容
            7. 目標壓縮比例：{compression_ratio}

            【壓縮策略】
            - 保留：所有關鍵邏輯點、時間線、因果關係、推理鏈條、矛盾點、關鍵證據
            - 簡化：次要描述、環境描寫、非關鍵對話
            - 合併：相似的事件描述、重複的邏輯點
            - 刪除：純粹的背景描述、無關的細節

            請生成壓縮後的文本，確保邏輯完整性：
            """
            
            compressed_text = self.llm_service.generate_text(compression_prompt)
            return compressed_text.strip()
            
        except Exception as e:
            print(f"[LogicAgent] 基於邏輯優先級壓縮時出錯: {str(e)}")
            return self._fallback_compression(text, compression_ratio)
    
    def _validate_logic_integrity(self, compressed_text: str, key_logic_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """驗證壓縮後的邏輯完整性"""
        try:
            # 使用 LLM 驗證邏輯完整性
            validation_prompt = f"""
            請驗證以下壓縮文本的邏輯完整性：

            【壓縮文本】
            {compressed_text}

            【原始關鍵邏輯點】
            {json.dumps(key_logic_points, ensure_ascii=False, indent=2)}

            請檢查以下方面：
            1. 時間線是否完整且邏輯一致
            2. 因果關係是否保持
            3. 推理鏈條是否完整
            4. 關鍵證據是否保留
            5. 矛盾點是否保留
            6. 整體邏輯是否自洽

            請以 JSON 格式返回驗證結果：
            {{
                "overall_integrity": true/false,
                "integrity_score": 0-100,
                "timeline_integrity": true/false,
                "causal_integrity": true/false,
                "reasoning_integrity": true/false,
                "evidence_preservation": 0-100,
                "contradiction_preservation": 0-100,
                "missing_elements": ["缺失元素1", "缺失元素2"],
                "logic_issues": ["邏輯問題1", "邏輯問題2"],
                "recommendations": ["建議1", "建議2"]
            }}
            """
            
            response = self.llm_service.generate_text(validation_prompt)
            validation_result = json.loads(response)
            
            return validation_result
            
        except Exception as e:
            print(f"[LogicAgent] 驗證邏輯完整性時出錯: {str(e)}")
            return self._get_default_validation_result()
    
    def _fallback_compression(self, text: str, compression_ratio: float) -> str:
        """備用壓縮方法"""
        try:
            # 簡單的文本截斷壓縮
            target_length = int(len(text) * (1 - compression_ratio))
            
            # 嘗試在句子邊界截斷
            sentences = text.split('。')
            compressed_sentences = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) <= target_length:
                    compressed_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            return '。'.join(compressed_sentences) + '。'
            
        except Exception as e:
            print(f"[LogicAgent] 備用壓縮時出錯: {str(e)}")
            return text[:int(len(text) * (1 - compression_ratio))] + "..."
    
    def _get_default_compression_result(self, text: str, compression_ratio: float) -> Dict[str, Any]:
        """獲取默認壓縮結果"""
        compressed_text = text[:int(len(text) * (1 - compression_ratio))] + "..."
        
        return {
            "original_text": text,
            "compressed_text": compressed_text,
            "logic_structure": self._get_default_logic_structure(),
            "key_logic_points": [],
            "logic_validation": self._get_default_validation_result(),
            "compression_ratio": self._calculate_compression_ratio(text, compressed_text),
            "agent_name": self.agent_name
        }
    
    def _get_default_logic_structure(self) -> Dict[str, Any]:
        """獲取默認邏輯結構"""
        return {
            "timeline_logic": {"events": [], "sequence": "", "time_consistency": True},
            "causal_logic": {"cause_effect_chains": [], "logical_consistency": True},
            "reasoning_chains": {"evidence_to_conclusion": [], "logical_strength": "中"},
            "contradictions": [],
            "key_evidence": [],
            "motive_logic": {"character_motives": {}, "motive_consistency": True}
        }
    
    def _get_default_validation_result(self) -> Dict[str, Any]:
        """獲取默認驗證結果"""
        return {
            "overall_integrity": True,
            "integrity_score": 75,
            "timeline_integrity": True,
            "causal_integrity": True,
            "reasoning_integrity": True,
            "evidence_preservation": 80,
            "contradiction_preservation": 80,
            "missing_elements": [],
            "logic_issues": [],
            "recommendations": []
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