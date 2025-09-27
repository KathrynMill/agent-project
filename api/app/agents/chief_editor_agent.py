"""
主編智能體 - 負責協調多智能體工作流程和 LangGraph 流程定義
作為整個壓縮流程的總控和決策中心
"""

from typing import Dict, List, Any, Optional, Annotated
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
import json
from datetime import datetime
from app.agents.analysis_agent import AnalysisAgent
from app.agents.logic_agent import LogicAgent
from app.agents.story_agent import StoryAgent
from app.agents.validation_agent import ValidationAgent
from app.services.nebula_service import NebulaService
from app.services.llm_service import LLMService


class CompressionState(TypedDict):
    """壓縮流程狀態定義"""
    # 輸入數據
    player_scripts: Dict[str, str]
    master_guide: str
    target_hours: int
    
    # 分析結果
    analysis_result: Optional[Dict[str, Any]]
    compression_strategy: Optional[Dict[str, Any]]
    
    # 壓縮結果
    logic_compression: Optional[Dict[str, Any]]
    story_compression: Optional[Dict[str, Any]]
    debate_result: Optional[Dict[str, Any]]
    final_compression: Optional[Dict[str, Any]]
    
    # 校驗結果
    validation_result: Optional[Dict[str, Any]]
    validation_passed: bool
    
    # 流程控制
    current_step: str
    iteration_count: int
    max_iterations: int
    error_messages: List[str]


class ChiefEditorAgent:
    """主編智能體：協調多智能體工作流程"""
    
    def __init__(self, nebula_service: NebulaService, llm_service: LLMService):
        self.nebula_service = nebula_service
        self.llm_service = llm_service
        
        # 初始化子智能體
        self.analysis_agent = AnalysisAgent(nebula_service, llm_service)
        self.logic_agent = LogicAgent(llm_service)
        self.story_agent = StoryAgent(llm_service)
        self.validation_agent = ValidationAgent(nebula_service, llm_service)
        
        self.agent_name = "ChiefEditorAgent"
    
    def create_compression_workflow(self):
        """創建 LangGraph 壓縮工作流程"""
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.prebuilt import ToolNode
            
            # 創建狀態圖
            workflow = StateGraph(CompressionState)
            
            # 添加節點
            workflow.add_node("analyze", self._analyze_step)
            workflow.add_node("plan", self._plan_step)
            workflow.add_node("logic_compress", self._logic_compress_step)
            workflow.add_node("story_compress", self._story_compress_step)
            workflow.add_node("debate", self._debate_step)
            workflow.add_node("integrate", self._integrate_step)
            workflow.add_node("validate", self._validate_step)
            workflow.add_node("finalize", self._finalize_step)
            
            # 定義流程邊
            workflow.set_entry_point("analyze")
            
            workflow.add_edge("analyze", "plan")
            workflow.add_edge("plan", "logic_compress")
            workflow.add_edge("plan", "story_compress")
            workflow.add_edge("logic_compress", "debate")
            workflow.add_edge("story_compress", "debate")
            workflow.add_edge("debate", "integrate")
            workflow.add_edge("integrate", "validate")
            
            # 條件邊：校驗結果決定下一步
            workflow.add_conditional_edges(
                "validate",
                self._should_continue,
                {
                    "continue": "logic_compress",  # 重新壓縮
                    "finalize": "finalize",
                    "end": END
                }
            )
            
            workflow.add_edge("finalize", END)
            
            # 編譯工作流程
            return workflow.compile()
            
        except ImportError:
            # 如果 LangGraph 不可用，使用簡化版本
            print("LangGraph not available, using simple workflow")
            return self._create_simple_workflow()
    
    def _create_simple_workflow(self):
        """創建簡化的工作流程（不依賴 LangGraph）"""
        return SimpleCompressionWorkflow(self)
    
    def _analyze_step(self, state: CompressionState) -> CompressionState:
        """分析步驟：分析圖譜結構"""
        try:
            print(f"[{self.agent_name}] 開始分析圖譜結構...")
            
            # 執行圖譜分析
            analysis_result = self.analysis_agent.analyze_graph_structure()
            
            state["analysis_result"] = analysis_result
            state["current_step"] = "analyze"
            
            print(f"[{self.agent_name}] 圖譜分析完成")
            return state
            
        except Exception as e:
            error_msg = f"分析步驟出錯: {str(e)}"
            state["error_messages"].append(error_msg)
            print(f"[{self.agent_name}] {error_msg}")
            return state
    
    def _plan_step(self, state: CompressionState) -> CompressionState:
        """規劃步驟：制定壓縮策略"""
        try:
            print(f"[{self.agent_name}] 開始制定壓縮策略...")
            
            analysis_result = state.get("analysis_result", {})
            compression_suggestions = analysis_result.get("compression_suggestions", {})
            
            # 制定詳細的壓縮策略
            strategy = {
                "target_compression_ratio": compression_suggestions.get("estimated_compression_ratio", "30-50%"),
                "character_merge_candidates": compression_suggestions.get("character_merge_suggestions", []),
                "event_simplification": compression_suggestions.get("event_simplification", []),
                "must_keep_elements": compression_suggestions.get("must_keep_contradictions", []),
                "compression_priority": compression_suggestions.get("compression_priority", []),
                "target_hours": state.get("target_hours", 4)
            }
            
            state["compression_strategy"] = strategy
            state["current_step"] = "plan"
            
            print(f"[{self.agent_name}] 壓縮策略制定完成")
            return state
            
        except Exception as e:
            error_msg = f"規劃步驟出錯: {str(e)}"
            state["error_messages"].append(error_msg)
            print(f"[{self.agent_name}] {error_msg}")
            return state
    
    def _logic_compress_step(self, state: CompressionState) -> CompressionState:
        """邏輯壓縮步驟"""
        try:
            print(f"[{self.agent_name}] 開始邏輯壓縮...")
            
            player_scripts = state.get("player_scripts", {})
            strategy = state.get("compression_strategy", {})
            
            # 合併所有劇本文本
            combined_text = self._combine_scripts(player_scripts)
            
            # 執行邏輯壓縮
            context = {
                "compression_suggestions": strategy,
                "analysis_result": state.get("analysis_result", {})
            }
            
            compression_result = self.logic_agent.compress_text(
                combined_text, context, 0.5
            )
            
            state["logic_compression"] = compression_result
            state["current_step"] = "logic_compress"
            
            print(f"[{self.agent_name}] 邏輯壓縮完成")
            return state
            
        except Exception as e:
            error_msg = f"邏輯壓縮步驟出錯: {str(e)}"
            state["error_messages"].append(error_msg)
            print(f"[{self.agent_name}] {error_msg}")
            return state
    
    def _story_compress_step(self, state: CompressionState) -> CompressionState:
        """故事壓縮步驟"""
        try:
            print(f"[{self.agent_name}] 開始故事壓縮...")
            
            player_scripts = state.get("player_scripts", {})
            strategy = state.get("compression_strategy", {})
            
            # 合併所有劇本文本
            combined_text = self._combine_scripts(player_scripts)
            
            # 執行故事壓縮
            context = {
                "compression_suggestions": strategy,
                "analysis_result": state.get("analysis_result", {})
            }
            
            compression_result = self.story_agent.compress_text(
                combined_text, context, 0.5
            )
            
            state["story_compression"] = compression_result
            state["current_step"] = "story_compress"
            
            print(f"[{self.agent_name}] 故事壓縮完成")
            return state
            
        except Exception as e:
            error_msg = f"故事壓縮步驟出錯: {str(e)}"
            state["error_messages"].append(error_msg)
            print(f"[{self.agent_name}] {error_msg}")
            return state
    
    def _debate_step(self, state: CompressionState) -> CompressionState:
        """辯論步驟：邏輯智能體和故事智能體辯論"""
        try:
            print(f"[{self.agent_name}] 開始智能體辯論...")
            
            logic_compression = state.get("logic_compression", {})
            story_compression = state.get("story_compression", {})
            
            if not logic_compression or not story_compression:
                state["debate_result"] = {"error": "缺少壓縮結果"}
                return state
            
            # 執行辯論
            logic_debate = self.logic_agent.debate_with_story_agent(
                logic_compression, story_compression
            )
            story_debate = self.story_agent.debate_with_logic_agent(
                story_compression, logic_compression
            )
            
            # 主編裁決
            final_decision = self._make_final_decision(
                logic_compression, story_compression, logic_debate, story_debate
            )
            
            state["debate_result"] = {
                "logic_debate": logic_debate,
                "story_debate": story_debate,
                "final_decision": final_decision
            }
            state["current_step"] = "debate"
            
            print(f"[{self.agent_name}] 辯論完成")
            return state
            
        except Exception as e:
            error_msg = f"辯論步驟出錯: {str(e)}"
            state["error_messages"].append(error_msg)
            print(f"[{self.agent_name}] {error_msg}")
            return state
    
    def _integrate_step(self, state: CompressionState) -> CompressionState:
        """整合步驟：整合壓縮結果"""
        try:
            print(f"[{self.agent_name}] 開始整合壓縮結果...")
            
            debate_result = state.get("debate_result", {})
            final_decision = debate_result.get("final_decision", {})
            
            # 根據決策整合結果
            if final_decision.get("decision") == "logic":
                base_compression = state.get("logic_compression", {})
            else:
                base_compression = state.get("story_compression", {})
            
            # 整合兩種壓縮的優點
            integrated_result = self._integrate_compressions(
                state.get("logic_compression", {}),
                state.get("story_compression", {}),
                final_decision
            )
            
            state["final_compression"] = integrated_result
            state["current_step"] = "integrate"
            
            print(f"[{self.agent_name}] 整合完成")
            return state
            
        except Exception as e:
            error_msg = f"整合步驟出錯: {str(e)}"
            state["error_messages"].append(error_msg)
            print(f"[{self.agent_name}] {error_msg}")
            return state
    
    def _validate_step(self, state: CompressionState) -> CompressionState:
        """校驗步驟：校驗壓縮結果"""
        try:
            print(f"[{self.agent_name}] 開始校驗壓縮結果...")
            
            final_compression = state.get("final_compression", {})
            compressed_text = final_compression.get("compressed_text", "")
            
            if not compressed_text:
                state["validation_passed"] = False
                state["validation_result"] = {"error": "沒有壓縮結果可校驗"}
                return state
            
            # 執行校驗
            validation_result = self.validation_agent.validate_compression(
                state.get("player_scripts", {}),
                compressed_text,
                state.get("master_guide", "")
            )
            
            state["validation_result"] = validation_result
            state["validation_passed"] = validation_result.get("overall_validation", {}).get("passes_validation", False)
            state["current_step"] = "validate"
            
            print(f"[{self.agent_name}] 校驗完成，結果: {'通過' if state['validation_passed'] else '未通過'}")
            return state
            
        except Exception as e:
            error_msg = f"校驗步驟出錯: {str(e)}"
            state["error_messages"].append(error_msg)
            print(f"[{self.agent_name}] {error_msg}")
            return state
    
    def _finalize_step(self, state: CompressionState) -> CompressionState:
        """完成步驟：生成最終結果"""
        try:
            print(f"[{self.agent_name}] 開始生成最終結果...")
            
            final_compression = state.get("final_compression", {})
            validation_result = state.get("validation_result", {})
            
            # 生成最終報告
            final_report = {
                "compressed_script": final_compression.get("compressed_text", ""),
                "compression_ratio": final_compression.get("compression_ratio", 0),
                "validation_summary": validation_result.get("overall_validation", {}),
                "agent_workflow": {
                    "analysis_result": state.get("analysis_result", {}),
                    "compression_strategy": state.get("compression_strategy", {}),
                    "debate_result": state.get("debate_result", {}),
                    "validation_result": validation_result
                },
                "metadata": {
                    "target_hours": state.get("target_hours", 4),
                    "iteration_count": state.get("iteration_count", 0),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            state["final_compression"] = final_report
            state["current_step"] = "finalize"
            
            print(f"[{self.agent_name}] 最終結果生成完成")
            return state
            
        except Exception as e:
            error_msg = f"完成步驟出錯: {str(e)}"
            state["error_messages"].append(error_msg)
            print(f"[{self.agent_name}] {error_msg}")
            return state
    
    def _should_continue(self, state: CompressionState) -> str:
        """決定是否繼續流程"""
        validation_passed = state.get("validation_passed", False)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        if validation_passed:
            return "finalize"
        elif iteration_count >= max_iterations:
            return "finalize"
        else:
            return "continue"
    
    def _make_final_decision(self, logic_compression: Dict, story_compression: Dict,
                                 logic_debate: Dict, story_debate: Dict) -> Dict[str, Any]:
        """主編做出最終決策"""
        
        decision_prompt = f"""
        作為主編，請基於兩個智能體的辯論結果做出最終決策：

        【邏輯智能體的方案】
        壓縮文本：{logic_compression.get('compressed_text', '')}
        邏輯驗證：{logic_compression.get('logic_validation', {})}
        辯論論點：{logic_debate.get('arguments', '')}

        【故事智能體的方案】
        壓縮文本：{story_compression.get('compressed_text', '')}
        故事驗證：{story_compression.get('story_validation', {})}
        辯論論點：{story_debate.get('arguments', '')}

        請做出決策並說明理由：
        1. 選擇哪個方案作為基礎？
        2. 需要整合哪些元素？
        3. 如何平衡邏輯完整性和故事吸引力？

        請以 JSON 格式返回：
        {{
            "decision": "logic" 或 "story",
            "reasoning": "決策理由",
            "integration_plan": ["整合計劃1", "整合計劃2"],
            "balance_strategy": "平衡策略"
        }}
        """
        
        try:
            response = self.llm_service.generate_text(decision_prompt)
            return json.loads(response)
        except:
            return {
                "decision": "logic",
                "reasoning": "邏輯完整性是推理的基礎",
                "integration_plan": ["整合故事細節", "增強戲劇效果"],
                "balance_strategy": "以邏輯為基礎，增強故事性"
            }
    
    def _integrate_compressions(self, logic_compression: Dict, story_compression: Dict,
                                    final_decision: Dict) -> Dict[str, Any]:
        """整合兩種壓縮結果"""
        
        integration_prompt = f"""
        請整合邏輯壓縮和故事壓縮的結果，創建最佳的壓縮版本：

        【邏輯壓縮結果】
        {logic_compression.get('compressed_text', '')}

        【故事壓縮結果】
        {story_compression.get('compressed_text', '')}

        【整合指導】
        決策：{final_decision.get('decision', 'logic')}
        理由：{final_decision.get('reasoning', '')}
        整合計劃：{final_decision.get('integration_plan', [])}

        請創建一個整合版本，平衡邏輯完整性和故事吸引力：
        """
        
        try:
            integrated_text = self.llm_service.generate_text(integration_prompt)
            
            return {
                "compressed_text": integrated_text,
                "compression_ratio": self._calculate_compression_ratio(
                    logic_compression.get('original_text', '') + story_compression.get('original_text', ''),
                    integrated_text
                ),
                "integration_method": "llm_integration",
                "base_decision": final_decision.get('decision', 'logic'),
                "agent_name": self.agent_name
            }
        except:
            # 如果整合失敗，返回決策選擇的版本
            if final_decision.get('decision') == 'logic':
                return logic_compression
            else:
                return story_compression
    
    def _combine_scripts(self, player_scripts: Dict[str, str]) -> str:
        """合併所有劇本"""
        combined = []
        for character, script in player_scripts.items():
            combined.append(f"【{character}的劇本】\n{script}\n")
        return "\n".join(combined)
    
    def _calculate_compression_ratio(self, original_text: str, compressed_text: str) -> float:
        """計算壓縮比例"""
        if not original_text:
            return 0.0
        
        original_length = len(original_text)
        compressed_length = len(compressed_text)
        
        if original_length == 0:
            return 0.0
        
        return round((original_length - compressed_length) / original_length, 3)


class SimpleCompressionWorkflow:
    """簡化的壓縮工作流程（不依賴 LangGraph）"""
    
    def __init__(self, chief_editor: ChiefEditorAgent):
        self.chief_editor = chief_editor
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """執行簡化工作流程"""
        
        # 初始化狀態
        state = CompressionState(
            player_scripts=input_data.get("player_scripts", {}),
            master_guide=input_data.get("master_guide", ""),
            target_hours=input_data.get("target_hours", 4),
            analysis_result=None,
            compression_strategy=None,
            logic_compression=None,
            story_compression=None,
            debate_result=None,
            final_compression=None,
            validation_result=None,
            validation_passed=False,
            current_step="start",
            iteration_count=0,
            max_iterations=3,
            error_messages=[]
        )
        
        # 執行流程步驟
        try:
            # 1. 分析
            state = self.chief_editor._analyze_step(state)
            
            # 2. 規劃
            state = self.chief_editor._plan_step(state)
            
            # 3. 壓縮（並行）
            state = self.chief_editor._logic_compress_step(state)
            state = self.chief_editor._story_compress_step(state)
            
            # 4. 辯論
            state = self.chief_editor._debate_step(state)
            
            # 5. 整合
            state = self.chief_editor._integrate_step(state)
            
            # 6. 校驗
            state = self.chief_editor._validate_step(state)
            
            # 7. 完成
            state = self.chief_editor._finalize_step(state)
            
            return state.get("final_compression", {})
            
        except Exception as e:
            return {
                "error": f"工作流程執行失敗: {str(e)}",
                "state": state
            }

