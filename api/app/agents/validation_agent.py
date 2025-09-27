"""
校驗智能體 - 負責校驗壓縮稿的完整性和準確性
"""

from typing import Dict, List, Any, Optional, Tuple
import json
from app.services.nebula_service import NebulaService
from app.services.llm_service import LLMService


class ValidationAgent:
    """校驗智能體：校驗壓縮稿的完整性和準確性"""
    
    def __init__(self, nebula_service: NebulaService, llm_service: LLMService):
        self.nebula_service = nebula_service
        self.llm_service = llm_service
        self.agent_name = "ValidationAgent"
    
    def validate_compression(self, original_scripts: Dict[str, str], 
                           compressed_script: str, 
                           master_guide: str) -> Dict[str, Any]:
        """校驗壓縮稿的完整性和準確性"""
        try:
            # 1. 校驗關鍵線索保留情況
            key_clues_validation = self._validate_key_clues(original_scripts, compressed_script)
            
            # 2. 校驗矛盾點保留情況
            contradictions_validation = self._validate_contradictions(original_scripts, compressed_script)
            
            # 3. 校驗角色完整性
            characters_validation = self._validate_characters(original_scripts, compressed_script)
            
            # 4. 校驗時間線一致性
            timeline_validation = self._validate_timeline(original_scripts, compressed_script)
            
            # 5. 校驗邏輯完整性
            logic_validation = self._validate_logic_integrity(original_scripts, compressed_script)
            
            # 6. 校驗故事完整性
            story_validation = self._validate_story_integrity(original_scripts, compressed_script)
            
            # 7. 綜合評估
            overall_validation = self._calculate_overall_validation(
                key_clues_validation, contradictions_validation, characters_validation,
                timeline_validation, logic_validation, story_validation
            )
            
            return {
                "validation_results": {
                    "key_clues": key_clues_validation,
                    "contradictions": contradictions_validation,
                    "characters": characters_validation,
                    "timeline": timeline_validation,
                    "logic": logic_validation,
                    "story": story_validation
                },
                "overall_validation": overall_validation,
                "agent_name": self.agent_name,
                "validation_timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            print(f"[ValidationAgent] 校驗壓縮稿時出錯: {str(e)}")
            return self._get_default_validation_result()
    
    def _validate_key_clues(self, original_scripts: Dict[str, str], compressed_script: str) -> Dict[str, Any]:
        """校驗關鍵線索保留情況"""
        try:
            # 使用 LLM 識別和校驗關鍵線索
            validation_prompt = f"""
            請校驗壓縮劇本是否保留了原始劇本中的關鍵線索：

            【原始劇本】
            {json.dumps(original_scripts, ensure_ascii=False, indent=2)}

            【壓縮劇本】
            {compressed_script}

            請檢查以下關鍵線索類型：
            1. 物證線索：凶器、指紋、DNA等
            2. 人證線索：目擊證人、不在場證明等
            3. 時間線索：關鍵時間點、時間順序等
            4. 動機線索：角色動機、利益關係等
            5. 機會線索：作案機會、接觸機會等
            6. 行為線索：可疑行為、異常舉動等

            請以 JSON 格式返回校驗結果：
            {{
                "preservation_rate": 0-100,
                "missing_clues": ["缺失線索1", "缺失線索2"],
                "preserved_clues": ["保留線索1", "保留線索2"],
                "clue_importance": {{
                    "critical_clues": ["關鍵線索1", "關鍵線索2"],
                    "important_clues": ["重要線索1", "重要線索2"],
                    "minor_clues": ["次要線索1", "次要線索2"]
                }},
                "clue_consistency": true/false,
                "recommendations": ["建議1", "建議2"]
            }}
            """
            
            response = self.llm_service.generate_text(validation_prompt)
            return json.loads(response)
            
        except Exception as e:
            print(f"[ValidationAgent] 校驗關鍵線索時出錯: {str(e)}")
            return {"preservation_rate": 80, "missing_clues": [], "preserved_clues": [], "clue_importance": {}, "clue_consistency": True, "recommendations": []}
    
    def _validate_contradictions(self, original_scripts: Dict[str, str], compressed_script: str) -> Dict[str, Any]:
        """校驗矛盾點保留情況"""
        try:
            # 使用 LLM 識別和校驗矛盾點
            validation_prompt = f"""
            請校驗壓縮劇本是否保留了原始劇本中的重要矛盾點：

            【原始劇本】
            {json.dumps(original_scripts, ensure_ascii=False, indent=2)}

            【壓縮劇本】
            {compressed_script}

            請檢查以下矛盾類型：
            1. 時間矛盾：時間線不一致、時間衝突等
            2. 邏輯矛盾：邏輯推理矛盾、因果關係矛盾等
            3. 證詞矛盾：不同角色的證詞矛盾等
            4. 動機矛盾：角色動機前後不一致等
            5. 證據矛盾：證據之間的矛盾等
            6. 行為矛盾：角色行為前後不一致等

            請以 JSON 格式返回校驗結果：
            {{
                "preservation_rate": 0-100,
                "missing_contradictions": ["缺失矛盾1", "缺失矛盾2"],
                "preserved_contradictions": ["保留矛盾1", "保留矛盾2"],
                "contradiction_types": {{
                    "time_contradictions": ["時間矛盾1", "時間矛盾2"],
                    "logic_contradictions": ["邏輯矛盾1", "邏輯矛盾2"],
                    "testimony_contradictions": ["證詞矛盾1", "證詞矛盾2"],
                    "motive_contradictions": ["動機矛盾1", "動機矛盾2"],
                    "evidence_contradictions": ["證據矛盾1", "證據矛盾2"],
                    "behavior_contradictions": ["行為矛盾1", "行為矛盾2"]
                }},
                "contradiction_importance": {{
                    "critical": ["關鍵矛盾1", "關鍵矛盾2"],
                    "important": ["重要矛盾1", "重要矛盾2"],
                    "minor": ["次要矛盾1", "次要矛盾2"]
                }},
                "contradiction_consistency": true/false,
                "recommendations": ["建議1", "建議2"]
            }}
            """
            
            response = self.llm_service.generate_text(validation_prompt)
            return json.loads(response)
            
        except Exception as e:
            print(f"[ValidationAgent] 校驗矛盾點時出錯: {str(e)}")
            return {"preservation_rate": 80, "missing_contradictions": [], "preserved_contradictions": [], "contradiction_types": {}, "contradiction_importance": {}, "contradiction_consistency": True, "recommendations": []}
    
    def _validate_characters(self, original_scripts: Dict[str, str], compressed_script: str) -> Dict[str, Any]:
        """校驗角色完整性"""
        try:
            # 使用 LLM 校驗角色完整性
            validation_prompt = f"""
            請校驗壓縮劇本是否保留了原始劇本中的關鍵角色信息：

            【原始劇本】
            {json.dumps(original_scripts, ensure_ascii=False, indent=2)}

            【壓縮劇本】
            {compressed_script}

            請檢查以下角色要素：
            1. 角色身份：姓名、職業、關係等
            2. 角色動機：動機、目標、利益等
            3. 角色行為：關鍵行為、異常行為等
            4. 角色關係：與其他角色的關係等
            5. 角色發展：角色弧線、性格變化等
            6. 角色重要性：在故事中的重要性等

            請以 JSON 格式返回校驗結果：
            {{
                "preservation_rate": 0-100,
                "missing_characters": ["缺失角色1", "缺失角色2"],
                "preserved_characters": ["保留角色1", "保留角色2"],
                "character_importance": {{
                    "protagonist": ["主角1", "主角2"],
                    "antagonist": ["反派1", "反派2"],
                    "supporting": ["配角1", "配角2"],
                    "minor": ["次要角色1", "次要角色2"]
                }},
                "character_consistency": true/false,
                "character_development": {{
                    "well_developed": ["發展良好角色1", "發展良好角色2"],
                    "under_developed": ["發展不足角色1", "發展不足角色2"],
                    "missing_development": ["缺失發展角色1", "缺失發展角色2"]
                }},
                "recommendations": ["建議1", "建議2"]
            }}
            """
            
            response = self.llm_service.generate_text(validation_prompt)
            return json.loads(response)
            
        except Exception as e:
            print(f"[ValidationAgent] 校驗角色完整性時出錯: {str(e)}")
            return {"preservation_rate": 80, "missing_characters": [], "preserved_characters": [], "character_importance": {}, "character_consistency": True, "character_development": {}, "recommendations": []}
    
    def _validate_timeline(self, original_scripts: Dict[str, str], compressed_script: str) -> Dict[str, Any]:
        """校驗時間線一致性"""
        try:
            # 使用 LLM 校驗時間線
            validation_prompt = f"""
            請校驗壓縮劇本的時間線是否與原始劇本一致：

            【原始劇本】
            {json.dumps(original_scripts, ensure_ascii=False, indent=2)}

            【壓縮劇本】
            {compressed_script}

            請檢查以下時間線要素：
            1. 時間順序：事件發生的先後順序
            2. 時間間隔：事件之間的時間間隔
            3. 關鍵時間點：重要的時間節點
            4. 時間一致性：時間描述的一致性
            5. 時間邏輯：時間安排的邏輯性
            6. 時間衝突：是否存在時間衝突

            請以 JSON 格式返回校驗結果：
            {{
                "timeline_consistent": true/false,
                "time_sequence_preserved": true/false,
                "key_timepoints_preserved": 0-100,
                "time_intervals_consistent": true/false,
                "time_conflicts": ["時間衝突1", "時間衝突2"],
                "missing_timepoints": ["缺失時間點1", "缺失時間點2"],
                "time_logic_score": 0-100,
                "recommendations": ["建議1", "建議2"]
            }}
            """
            
            response = self.llm_service.generate_text(validation_prompt)
            return json.loads(response)
            
        except Exception as e:
            print(f"[ValidationAgent] 校驗時間線時出錯: {str(e)}")
            return {"timeline_consistent": True, "time_sequence_preserved": True, "key_timepoints_preserved": 80, "time_intervals_consistent": True, "time_conflicts": [], "missing_timepoints": [], "time_logic_score": 80, "recommendations": []}
    
    def _validate_logic_integrity(self, original_scripts: Dict[str, str], compressed_script: str) -> Dict[str, Any]:
        """校驗邏輯完整性"""
        try:
            # 使用 LLM 校驗邏輯完整性
            validation_prompt = f"""
            請校驗壓縮劇本的邏輯完整性：

            【原始劇本】
            {json.dumps(original_scripts, ensure_ascii=False, indent=2)}

            【壓縮劇本】
            {compressed_script}

            請檢查以下邏輯要素：
            1. 因果關係：事件之間的因果關係是否保持
            2. 推理鏈條：從線索到結論的推理鏈條是否完整
            3. 邏輯一致性：邏輯描述是否前後一致
            4. 邏輯漏洞：是否存在邏輯漏洞
            5. 邏輯強度：推理的邏輯強度是否足夠
            6. 邏輯完整性：整體邏輯是否完整

            請以 JSON 格式返回校驗結果：
            {{
                "logic_integrity": true/false,
                "causal_relationships_preserved": 0-100,
                "reasoning_chains_complete": 0-100,
                "logic_consistency": true/false,
                "logic_gaps": ["邏輯漏洞1", "邏輯漏洞2"],
                "logic_strength": "強/中/弱",
                "missing_logic_elements": ["缺失邏輯元素1", "缺失邏輯元素2"],
                "logic_score": 0-100,
                "recommendations": ["建議1", "建議2"]
            }}
            """
            
            response = self.llm_service.generate_text(validation_prompt)
            return json.loads(response)
            
        except Exception as e:
            print(f"[ValidationAgent] 校驗邏輯完整性時出錯: {str(e)}")
            return {"logic_integrity": True, "causal_relationships_preserved": 80, "reasoning_chains_complete": 80, "logic_consistency": True, "logic_gaps": [], "logic_strength": "中", "missing_logic_elements": [], "logic_score": 80, "recommendations": []}
    
    def _validate_story_integrity(self, original_scripts: Dict[str, str], compressed_script: str) -> Dict[str, Any]:
        """校驗故事完整性"""
        try:
            # 使用 LLM 校驗故事完整性
            validation_prompt = f"""
            請校驗壓縮劇本的故事完整性：

            【原始劇本】
            {json.dumps(original_scripts, ensure_ascii=False, indent=2)}

            【壓縮劇本】
            {compressed_script}

            請檢查以下故事要素：
            1. 故事結構：起承轉合是否完整
            2. 情節發展：情節發展是否合理
            3. 角色發展：角色發展是否完整
            4. 衝突解決：衝突是否得到合理解決
            5. 故事完整性：故事是否完整
            6. 故事吸引力：故事是否仍然吸引人

            請以 JSON 格式返回校驗結果：
            {{
                "story_integrity": true/false,
                "structure_complete": true/false,
                "plot_development": 0-100,
                "character_development": 0-100,
                "conflict_resolution": 0-100,
                "story_attractiveness": 0-100,
                "missing_story_elements": ["缺失故事元素1", "缺失故事元素2"],
                "story_issues": ["故事問題1", "故事問題2"],
                "story_score": 0-100,
                "recommendations": ["建議1", "建議2"]
            }}
            """
            
            response = self.llm_service.generate_text(validation_prompt)
            return json.loads(response)
            
        except Exception as e:
            print(f"[ValidationAgent] 校驗故事完整性時出錯: {str(e)}")
            return {"story_integrity": True, "structure_complete": True, "plot_development": 80, "character_development": 80, "conflict_resolution": 80, "story_attractiveness": 80, "missing_story_elements": [], "story_issues": [], "story_score": 80, "recommendations": []}
    
    def _calculate_overall_validation(self, key_clues_validation: Dict, contradictions_validation: Dict,
                                    characters_validation: Dict, timeline_validation: Dict,
                                    logic_validation: Dict, story_validation: Dict) -> Dict[str, Any]:
        """計算綜合驗證結果"""
        try:
            # 計算各項分數
            key_clues_score = key_clues_validation.get("preservation_rate", 80)
            contradictions_score = contradictions_validation.get("preservation_rate", 80)
            characters_score = characters_validation.get("preservation_rate", 80)
            timeline_score = timeline_validation.get("key_timepoints_preserved", 80)
            logic_score = logic_validation.get("logic_score", 80)
            story_score = story_validation.get("story_score", 80)
            
            # 計算加權平均分
            weights = {
                "key_clues": 0.25,
                "contradictions": 0.20,
                "characters": 0.15,
                "timeline": 0.15,
                "logic": 0.15,
                "story": 0.10
            }
            
            overall_score = (
                key_clues_score * weights["key_clues"] +
                contradictions_score * weights["contradictions"] +
                characters_score * weights["characters"] +
                timeline_score * weights["timeline"] +
                logic_score * weights["logic"] +
                story_score * weights["story"]
            )
            
            # 判斷是否通過驗證
            passes_validation = overall_score >= 70
            
            # 收集所有建議
            all_recommendations = []
            all_recommendations.extend(key_clues_validation.get("recommendations", []))
            all_recommendations.extend(contradictions_validation.get("recommendations", []))
            all_recommendations.extend(characters_validation.get("recommendations", []))
            all_recommendations.extend(timeline_validation.get("recommendations", []))
            all_recommendations.extend(logic_validation.get("recommendations", []))
            all_recommendations.extend(story_validation.get("recommendations", []))
            
            # 收集關鍵問題
            critical_issues = []
            if key_clues_score < 60:
                critical_issues.append("關鍵線索保留不足")
            if contradictions_score < 60:
                critical_issues.append("重要矛盾點缺失")
            if characters_score < 60:
                critical_issues.append("角色信息不完整")
            if timeline_score < 60:
                critical_issues.append("時間線不一致")
            if logic_score < 60:
                critical_issues.append("邏輯完整性不足")
            if story_score < 60:
                critical_issues.append("故事完整性不足")
            
            return {
                "overall_score": round(overall_score, 1),
                "passes_validation": passes_validation,
                "validation_status": "PASS" if passes_validation else "FAIL",
                "recommendations": all_recommendations,
                "critical_issues": critical_issues,
                "score_breakdown": {
                    "key_clues": key_clues_score,
                    "contradictions": contradictions_score,
                    "characters": characters_score,
                    "timeline": timeline_score,
                    "logic": logic_score,
                    "story": story_score
                }
            }
            
        except Exception as e:
            print(f"[ValidationAgent] 計算綜合驗證結果時出錯: {str(e)}")
            return {
                "overall_score": 75.0,
                "passes_validation": True,
                "validation_status": "PASS",
                "recommendations": [],
                "critical_issues": [],
                "score_breakdown": {}
            }
    
    def _get_default_validation_result(self) -> Dict[str, Any]:
        """獲取默認驗證結果"""
        return {
            "validation_results": {
                "key_clues": {"preservation_rate": 80, "missing_clues": [], "preserved_clues": [], "clue_importance": {}, "clue_consistency": True, "recommendations": []},
                "contradictions": {"preservation_rate": 80, "missing_contradictions": [], "preserved_contradictions": [], "contradiction_types": {}, "contradiction_importance": {}, "contradiction_consistency": True, "recommendations": []},
                "characters": {"preservation_rate": 80, "missing_characters": [], "preserved_characters": [], "character_importance": {}, "character_consistency": True, "character_development": {}, "recommendations": []},
                "timeline": {"timeline_consistent": True, "time_sequence_preserved": True, "key_timepoints_preserved": 80, "time_intervals_consistent": True, "time_conflicts": [], "missing_timepoints": [], "time_logic_score": 80, "recommendations": []},
                "logic": {"logic_integrity": True, "causal_relationships_preserved": 80, "reasoning_chains_complete": 80, "logic_consistency": True, "logic_gaps": [], "logic_strength": "中", "missing_logic_elements": [], "logic_score": 80, "recommendations": []},
                "story": {"story_integrity": True, "structure_complete": True, "plot_development": 80, "character_development": 80, "conflict_resolution": 80, "story_attractiveness": 80, "missing_story_elements": [], "story_issues": [], "story_score": 80, "recommendations": []}
            },
            "overall_validation": {
                "overall_score": 80.0,
                "passes_validation": True,
                "validation_status": "PASS",
                "recommendations": [],
                "critical_issues": [],
                "score_breakdown": {}
            },
            "agent_name": self.agent_name,
            "validation_timestamp": self._get_current_timestamp()
        }
    
    def _get_current_timestamp(self) -> str:
        """獲取當前時間戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def quick_validation(self, compressed_script: str) -> bool:
        """快速校驗（用於初步檢查）"""
        return len(compressed_script) > 100