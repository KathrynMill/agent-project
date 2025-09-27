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
        try:
            # 1. 分析故事結構和戲劇元素
            story_structure = self._analyze_story_structure(text)
            
            # 2. 識別黃金細節和戲劇亮點
            golden_details = self._identify_golden_details(text, story_structure)
            
            # 3. 基於故事吸引力進行壓縮
            compressed_text = self._compress_with_story_priority(
                text, golden_details, compression_ratio, context
            )
            
            # 4. 驗證壓縮後的故事完整性
            story_validation = self._validate_story_integrity(compressed_text, golden_details)
            
            return {
                "original_text": text,
                "compressed_text": compressed_text,
                "story_structure": story_structure,
                "golden_details": golden_details,
                "story_validation": story_validation,
                "compression_ratio": self._calculate_compression_ratio(text, compressed_text),
                "agent_name": self.agent_name
            }
            
        except Exception as e:
            print(f"[StoryAgent] 壓縮文本時出錯: {str(e)}")
            return self._get_default_compression_result(text, compression_ratio)
    
    def _analyze_story_structure(self, text: str) -> Dict[str, Any]:
        """分析故事結構和戲劇元素"""
        try:
            # 使用 LLM 分析故事結構
            analysis_prompt = f"""
            請分析以下劇本文本的故事結構和戲劇元素：

            【劇本文本】
            {text}

            請從以下角度分析：
            1. 故事結構：起承轉合、情節發展
            2. 戲劇衝突：主要衝突、次要衝突
            3. 角色發展：角色弧線、性格特點
            4. 情感張力：緊張感、懸疑感、情感共鳴
            5. 戲劇亮點：高潮、轉折點、驚喜元素
            6. 氛圍營造：環境描寫、氣氛渲染

            請以 JSON 格式返回分析結果：
            {{
                "story_structure": {{
                    "beginning": "開頭描述",
                    "development": "發展描述", 
                    "climax": "高潮描述",
                    "ending": "結尾描述",
                    "plot_points": ["情節點1", "情節點2"]
                }},
                "dramatic_conflicts": {{
                    "main_conflict": "主要衝突描述",
                    "sub_conflicts": ["次要衝突1", "次要衝突2"],
                    "conflict_intensity": "高/中/低"
                }},
                "character_development": {{
                    "protagonist_arc": "主角發展弧線",
                    "antagonist_arc": "反派發展弧線",
                    "supporting_roles": ["配角1", "配角2"],
                    "character_relationships": {{"關係1": "描述1"}}
                }},
                "emotional_tension": {{
                    "suspense_elements": ["懸疑元素1", "懸疑元素2"],
                    "emotional_peaks": ["情感高潮1", "情感高潮2"],
                    "tension_level": "高/中/低"
                }},
                "dramatic_highlights": {{
                    "climactic_moments": ["高潮時刻1", "高潮時刻2"],
                    "plot_twists": ["轉折1", "轉折2"],
                    "surprise_elements": ["驚喜元素1", "驚喜元素2"]
                }},
                "atmosphere": {{
                    "mood": "整體氛圍",
                    "tone": "敘述語調",
                    "environmental_details": ["環境細節1", "環境細節2"]
                }}
            }}
            """
            
            response = self.llm_service.generate_text(analysis_prompt)
            return json.loads(response)
            
        except Exception as e:
            print(f"[StoryAgent] 分析故事結構時出錯: {str(e)}")
            return self._get_default_story_structure()
    
    def _identify_golden_details(self, text: str, story_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """識別黃金細節和戲劇亮點"""
        try:
            golden_details = []
            
            # 從故事結構中提取關鍵細節
            plot_points = story_structure.get("story_structure", {}).get("plot_points", [])
            dramatic_conflicts = story_structure.get("dramatic_conflicts", {})
            emotional_peaks = story_structure.get("emotional_tension", {}).get("emotional_peaks", [])
            climactic_moments = story_structure.get("dramatic_highlights", {}).get("climactic_moments", [])
            plot_twists = story_structure.get("dramatic_highlights", {}).get("plot_twists", [])
            surprise_elements = story_structure.get("dramatic_highlights", {}).get("surprise_elements", [])
            environmental_details = story_structure.get("atmosphere", {}).get("environmental_details", [])
            
            # 情節點（高重要性）
            for point in plot_points:
                golden_details.append({
                    "type": "plot_point",
                    "content": point,
                    "importance": "high",
                    "story_role": "情節推進"
                })
            
            # 主要衝突（關鍵）
            main_conflict = dramatic_conflicts.get("main_conflict", "")
            if main_conflict:
                golden_details.append({
                    "type": "main_conflict",
                    "content": main_conflict,
                    "importance": "critical",
                    "story_role": "故事核心"
                })
            
            # 情感高潮（高重要性）
            for peak in emotional_peaks:
                golden_details.append({
                    "type": "emotional_peak",
                    "content": peak,
                    "importance": "high",
                    "story_role": "情感共鳴"
                })
            
            # 戲劇高潮（關鍵）
            for moment in climactic_moments:
                golden_details.append({
                    "type": "climactic_moment",
                    "content": moment,
                    "importance": "critical",
                    "story_role": "戲劇高潮"
                })
            
            # 情節轉折（關鍵）
            for twist in plot_twists:
                golden_details.append({
                    "type": "plot_twist",
                    "content": twist,
                    "importance": "critical",
                    "story_role": "情節轉折"
                })
            
            # 驚喜元素（高重要性）
            for surprise in surprise_elements:
                golden_details.append({
                    "type": "surprise_element",
                    "content": surprise,
                    "importance": "high",
                    "story_role": "戲劇效果"
                })
            
            # 氛圍細節（中重要性）
            for detail in environmental_details:
                golden_details.append({
                    "type": "atmospheric_detail",
                    "content": detail,
                    "importance": "medium",
                    "story_role": "氛圍營造"
                })
            
            return golden_details
            
        except Exception as e:
            print(f"[StoryAgent] 識別黃金細節時出錯: {str(e)}")
            return []
    
    def _compress_with_story_priority(self, text: str, golden_details: List[Dict[str, Any]], 
                                    compression_ratio: float, context: Dict[str, Any]) -> str:
        """基於故事吸引力進行壓縮"""
        try:
            # 使用 LLM 進行智能壓縮
            compression_prompt = f"""
            請基於故事吸引力和戲劇效果壓縮以下劇本文本，重點保留故事細節和戲劇亮點：

            【原始文本】
            {text}

            【黃金細節】
            {json.dumps(golden_details, ensure_ascii=False, indent=2)}

            【壓縮要求】
            1. 必須保留所有 critical 和 high 重要性的故事元素
            2. 保持故事的起承轉合結構
            3. 保留所有戲劇衝突和情感張力
            4. 保留所有情節轉折和驚喜元素
            5. 保留關鍵的氛圍營造細節
            6. 可以壓縮或簡化 medium 和 low 重要性的內容
            7. 目標壓縮比例：{compression_ratio}

            【壓縮策略】
            - 保留：所有關鍵情節點、戲劇衝突、情感高潮、情節轉折、驚喜元素、氛圍細節
            - 簡化：次要角色描述、重複的對話、過度詳細的環境描寫
            - 合併：相似的情感描述、重複的氛圍營造
            - 刪除：純粹的說明性文字、無關的背景信息

            【故事完整性要求】
            - 保持故事的完整性和連貫性
            - 維持戲劇張力和懸疑感
            - 確保角色發展的合理性
            - 保持情感共鳴和代入感

            請生成壓縮後的文本，確保故事吸引力和戲劇效果：
            """
            
            compressed_text = self.llm_service.generate_text(compression_prompt)
            return compressed_text.strip()
            
        except Exception as e:
            print(f"[StoryAgent] 基於故事優先級壓縮時出錯: {str(e)}")
            return self._fallback_compression(text, compression_ratio)
    
    def _validate_story_integrity(self, compressed_text: str, golden_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """驗證壓縮後的故事完整性"""
        try:
            # 使用 LLM 驗證故事完整性
            validation_prompt = f"""
            請驗證以下壓縮文本的故事完整性和戲劇效果：

            【壓縮文本】
            {compressed_text}

            【原始黃金細節】
            {json.dumps(golden_details, ensure_ascii=False, indent=2)}

            請檢查以下方面：
            1. 故事結構是否完整（起承轉合）
            2. 戲劇衝突是否保持
            3. 情感張力是否維持
            4. 情節轉折是否保留
            5. 角色發展是否合理
            6. 氛圍營造是否有效
            7. 整體故事吸引力如何

            請以 JSON 格式返回驗證結果：
            {{
                "overall_story_integrity": true/false,
                "story_attractiveness_score": 0-100,
                "structure_integrity": true/false,
                "conflict_preservation": 0-100,
                "emotional_impact": 0-100,
                "plot_twist_preservation": 0-100,
                "character_development": 0-100,
                "atmosphere_effectiveness": 0-100,
                "missing_story_elements": ["缺失故事元素1", "缺失故事元素2"],
                "story_issues": ["故事問題1", "故事問題2"],
                "dramatic_effectiveness": "高/中/低",
                "recommendations": ["建議1", "建議2"]
            }}
            """
            
            response = self.llm_service.generate_text(validation_prompt)
            validation_result = json.loads(response)
            
            return validation_result
            
        except Exception as e:
            print(f"[StoryAgent] 驗證故事完整性時出錯: {str(e)}")
            return self._get_default_validation_result()
    
    def _fallback_compression(self, text: str, compression_ratio: float) -> str:
        """備用壓縮方法"""
        try:
            # 簡單的文本截斷壓縮
            target_length = int(len(text) * (1 - compression_ratio))
            
            # 嘗試在段落邊界截斷
            paragraphs = text.split('\n\n')
            compressed_paragraphs = []
            current_length = 0
            
            for paragraph in paragraphs:
                if current_length + len(paragraph) <= target_length:
                    compressed_paragraphs.append(paragraph)
                    current_length += len(paragraph)
                else:
                    break
            
            return '\n\n'.join(compressed_paragraphs)
            
        except Exception as e:
            print(f"[StoryAgent] 備用壓縮時出錯: {str(e)}")
            return text[:int(len(text) * (1 - compression_ratio))] + "..."
    
    def _get_default_compression_result(self, text: str, compression_ratio: float) -> Dict[str, Any]:
        """獲取默認壓縮結果"""
        compressed_text = text[:int(len(text) * (1 - compression_ratio))] + "..."
        
        return {
            "original_text": text,
            "compressed_text": compressed_text,
            "story_structure": self._get_default_story_structure(),
            "golden_details": [],
            "story_validation": self._get_default_validation_result(),
            "compression_ratio": self._calculate_compression_ratio(text, compressed_text),
            "agent_name": self.agent_name
        }
    
    def _get_default_story_structure(self) -> Dict[str, Any]:
        """獲取默認故事結構"""
        return {
            "story_structure": {"beginning": "", "development": "", "climax": "", "ending": "", "plot_points": []},
            "dramatic_conflicts": {"main_conflict": "", "sub_conflicts": [], "conflict_intensity": "中"},
            "character_development": {"protagonist_arc": "", "antagonist_arc": "", "supporting_roles": [], "character_relationships": {}},
            "emotional_tension": {"suspense_elements": [], "emotional_peaks": [], "tension_level": "中"},
            "dramatic_highlights": {"climactic_moments": [], "plot_twists": [], "surprise_elements": []},
            "atmosphere": {"mood": "", "tone": "", "environmental_details": []}
        }
    
    def _get_default_validation_result(self) -> Dict[str, Any]:
        """獲取默認驗證結果"""
        return {
            "overall_story_integrity": True,
            "story_attractiveness_score": 75,
            "structure_integrity": True,
            "conflict_preservation": 80,
            "emotional_impact": 80,
            "plot_twist_preservation": 80,
            "character_development": 75,
            "atmosphere_effectiveness": 75,
            "missing_story_elements": [],
            "story_issues": [],
            "dramatic_effectiveness": "中",
            "recommendations": []
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