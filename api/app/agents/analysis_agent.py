"""
分析智能體 - 負責分析全局圖譜並生成壓縮策略報告
"""

from typing import Dict, List, Any, Optional
import json
from app.services.nebula_service import NebulaService
from app.services.llm_service import LLMService


class AnalysisAgent:
    """分析智能體：分析圖譜結構，生成壓縮策略報告"""
    
    def __init__(self, nebula_service: NebulaService, llm_service: LLMService):
        self.nebula_service = nebula_service
        self.llm_service = llm_service
    
    def analyze_graph_structure(self) -> Dict[str, Any]:
        """分析圖譜結構，生成壓縮策略報告"""
        try:
            # 1. 獲取圖譜統計信息
            graph_stats = self._get_graph_statistics()
            
            # 2. 分析核心元素
            core_elements = self._analyze_core_elements()
            
            # 3. 分析矛盾點和陷阱
            trick_analysis = self._analyze_tricks_and_contradictions()
            
            # 4. 生成壓縮建議
            compression_suggestions = self._generate_compression_suggestions(
                graph_stats, core_elements, trick_analysis
            )
            
            return {
                "graph_statistics": graph_stats,
                "core_elements": core_elements,
                "trick_analysis": trick_analysis,
                "compression_suggestions": compression_suggestions,
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            print(f"[AnalysisAgent] 分析圖譜結構時出錯: {str(e)}")
            return self._get_default_analysis_result()
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """獲取圖譜統計信息"""
        try:
            # 查詢節點統計
            node_stats = self._query_node_statistics()
            
            # 查詢邊統計
            edge_stats = self._query_edge_statistics()
            
            # 查詢角色重要性分佈
            character_importance = self._analyze_character_importance()
            
            return {
                "nodes": node_stats,
                "edges": edge_stats,
                "character_importance": character_importance,
                "graph_density": self._calculate_graph_density(node_stats, edge_stats)
            }
            
        except Exception as e:
            print(f"[AnalysisAgent] 獲取圖譜統計時出錯: {str(e)}")
            return {"nodes": {"total": 0}, "edges": {"total": 0}}
    
    def _query_node_statistics(self) -> Dict[str, Any]:
        """查詢節點統計信息"""
        try:
            # 查詢各類型節點數量
            person_count = self.nebula_service.execute_query(
                "MATCH (n:Person) RETURN count(n) as count"
            )
            location_count = self.nebula_service.execute_query(
                "MATCH (n:Location) RETURN count(n) as count"
            )
            event_count = self.nebula_service.execute_query(
                "MATCH (n:Event) RETURN count(n) as count"
            )
            item_count = self.nebula_service.execute_query(
                "MATCH (n:Item) RETURN count(n) as count"
            )
            
            return {
                "total": person_count[0].get("count", 0) + 
                        location_count[0].get("count", 0) + 
                        event_count[0].get("count", 0) + 
                        item_count[0].get("count", 0),
                "by_type": {
                    "Person": person_count[0].get("count", 0),
                    "Location": location_count[0].get("count", 0),
                    "Event": event_count[0].get("count", 0),
                    "Item": item_count[0].get("count", 0)
                }
            }
            
        except Exception as e:
            print(f"[AnalysisAgent] 查詢節點統計時出錯: {str(e)}")
            return {"total": 0, "by_type": {}}
    
    def _query_edge_statistics(self) -> Dict[str, Any]:
        """查詢邊統計信息"""
        try:
            # 查詢各類型邊數量
            edge_types = ["LOCATED_IN", "HAS_ITEM", "PARTICIPATED_IN", "HAPPENED_AT", "HAPPENED_ON"]
            edge_counts = {}
            total_edges = 0
            
            for edge_type in edge_types:
                result = self.nebula_service.execute_query(
                    f"MATCH ()-[r:{edge_type}]->() RETURN count(r) as count"
                )
                count = result[0].get("count", 0) if result else 0
                edge_counts[edge_type] = count
                total_edges += count
            
            return {
                "total": total_edges,
                "by_type": edge_counts
            }
            
        except Exception as e:
            print(f"[AnalysisAgent] 查詢邊統計時出錯: {str(e)}")
            return {"total": 0, "by_type": {}}
    
    def _analyze_character_importance(self) -> Dict[str, Any]:
        """分析角色重要性"""
        try:
            # 查詢角色連接度
            character_connections = self.nebula_service.execute_query(
                """
                MATCH (p:Person)-[r]-()
                RETURN p.name as name, count(r) as connection_count
                ORDER BY connection_count DESC
                """
            )
            
            # 查詢角色參與的事件數量
            character_events = self.nebula_service.execute_query(
                """
                MATCH (p:Person)-[:PARTICIPATED_IN]->(e:Event)
                RETURN p.name as name, count(e) as event_count
                ORDER BY event_count DESC
                """
            )
            
            # 計算重要性分數
            importance_scores = {}
            for char in character_connections:
                name = char.get("name", "")
                connections = char.get("connection_count", 0)
                events = next((e.get("event_count", 0) for e in character_events if e.get("name") == name), 0)
                
                # 重要性分數 = 連接度 * 0.6 + 事件參與度 * 0.4
                importance_scores[name] = connections * 0.6 + events * 0.4
            
            return {
                "scores": importance_scores,
                "top_characters": sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            print(f"[AnalysisAgent] 分析角色重要性時出錯: {str(e)}")
            return {"scores": {}, "top_characters": []}
    
    def _analyze_core_elements(self) -> Dict[str, Any]:
        """分析核心元素"""
        try:
            # 分析關鍵角色
            key_characters = self._identify_key_characters()
            
            # 分析關鍵事件
            key_events = self._identify_key_events()
            
            # 分析關鍵地點
            key_locations = self._identify_key_locations()
            
            # 分析關鍵物品
            key_items = self._identify_key_items()
            
            return {
                "characters": key_characters,
                "events": key_events,
                "locations": key_locations,
                "items": key_items
            }
            
        except Exception as e:
            print(f"[AnalysisAgent] 分析核心元素時出錯: {str(e)}")
            return {"characters": [], "events": [], "locations": [], "items": []}
    
    def _identify_key_characters(self) -> List[Dict[str, Any]]:
        """識別關鍵角色"""
        try:
            # 查詢角色詳細信息
            characters = self.nebula_service.execute_query(
                """
                MATCH (p:Person)
                OPTIONAL MATCH (p)-[:PARTICIPATED_IN]->(e:Event)
                OPTIONAL MATCH (p)-[r]-()
                RETURN p.name as name, p.role as role, 
                       count(e) as event_count, count(r) as total_connections
                ORDER BY total_connections DESC, event_count DESC
                """
            )
            
            key_characters = []
            for char in characters:
                name = char.get("name", "")
                role = char.get("role", "")
                event_count = char.get("event_count", 0)
                connections = char.get("total_connections", 0)
                
                # 判斷是否為關鍵角色
                is_key = (event_count >= 2 or connections >= 3 or 
                         role in ["主角", "嫌疑人", "受害者", "偵探"])
                
                key_characters.append({
                    "name": name,
                    "role": role,
                    "event_count": event_count,
                    "connections": connections,
                    "is_key": is_key,
                    "importance_score": event_count * 0.6 + connections * 0.4
                })
            
            return key_characters
            
        except Exception as e:
            print(f"[AnalysisAgent] 識別關鍵角色時出錯: {str(e)}")
            return []
    
    def _identify_key_events(self) -> List[Dict[str, Any]]:
        """識別關鍵事件"""
        try:
            # 查詢事件詳細信息
            events = self.nebula_service.execute_query(
                """
                MATCH (e:Event)
                OPTIONAL MATCH (p:Person)-[:PARTICIPATED_IN]->(e)
                RETURN e.name as name, e.description as description,
                       count(p) as participant_count
                ORDER BY participant_count DESC
                """
            )
            
            key_events = []
            for event in events:
                name = event.get("name", "")
                description = event.get("description", "")
                participants = event.get("participant_count", 0)
                
                # 判斷是否為關鍵事件
                is_key = (participants >= 2 or 
                         any(keyword in description for keyword in ["死亡", "謀殺", "發現", "爭吵", "秘密"]))
                
                key_events.append({
                    "name": name,
                    "description": description,
                    "participant_count": participants,
                    "is_key": is_key,
                    "importance_score": participants * 0.5 + (1 if is_key else 0) * 0.5
                })
            
            return key_events
            
        except Exception as e:
            print(f"[AnalysisAgent] 識別關鍵事件時出錯: {str(e)}")
            return []
    
    def _identify_key_locations(self) -> List[Dict[str, Any]]:
        """識別關鍵地點"""
        try:
            # 查詢地點詳細信息
            locations = self.nebula_service.execute_query(
                """
                MATCH (l:Location)
                OPTIONAL MATCH (p:Person)-[:LOCATED_IN]->(l)
                OPTIONAL MATCH (e:Event)-[:HAPPENED_AT]->(l)
                RETURN l.name as name, count(p) as person_count, count(e) as event_count
                ORDER BY event_count DESC, person_count DESC
                """
            )
            
            key_locations = []
            for loc in locations:
                name = loc.get("name", "")
                person_count = loc.get("person_count", 0)
                event_count = loc.get("event_count", 0)
                
                # 判斷是否為關鍵地點
                is_key = (event_count >= 1 or person_count >= 2)
                
                key_locations.append({
                    "name": name,
                    "person_count": person_count,
                    "event_count": event_count,
                    "is_key": is_key,
                    "importance_score": event_count * 0.7 + person_count * 0.3
                })
            
            return key_locations
            
        except Exception as e:
            print(f"[AnalysisAgent] 識別關鍵地點時出錯: {str(e)}")
            return []
    
    def _identify_key_items(self) -> List[Dict[str, Any]]:
        """識別關鍵物品"""
        try:
            # 查詢物品詳細信息
            items = self.nebula_service.execute_query(
                """
                MATCH (i:Item)
                OPTIONAL MATCH (p:Person)-[:HAS_ITEM]->(i)
                OPTIONAL MATCH (e:Event)-[:HAPPENED_AT]->(i)
                RETURN i.name as name, i.type as type, count(p) as owner_count, count(e) as event_count
                ORDER BY event_count DESC, owner_count DESC
                """
            )
            
            key_items = []
            for item in items:
                name = item.get("name", "")
                item_type = item.get("type", "")
                owner_count = item.get("owner_count", 0)
                event_count = item.get("event_count", 0)
                
                # 判斷是否為關鍵物品
                is_key = (event_count >= 1 or owner_count >= 1 or 
                         any(keyword in name for keyword in ["刀", "毒", "鑰匙", "文件", "手機"]))
                
                key_items.append({
                    "name": name,
                    "type": item_type,
                    "owner_count": owner_count,
                    "event_count": event_count,
                    "is_key": is_key,
                    "importance_score": event_count * 0.6 + owner_count * 0.4
                })
            
            return key_items
            
        except Exception as e:
            print(f"[AnalysisAgent] 識別關鍵物品時出錯: {str(e)}")
            return []
    
    def _analyze_tricks_and_contradictions(self) -> Dict[str, Any]:
        """分析陷阱和矛盾"""
        try:
            # 分析時間線矛盾
            timeline_contradictions = self._analyze_timeline_contradictions()
            
            # 分析不在場證明矛盾
            alibi_contradictions = self._analyze_alibi_contradictions()
            
            # 分析動機矛盾
            motive_contradictions = self._analyze_motive_contradictions()
            
            # 分析證據矛盾
            evidence_contradictions = self._analyze_evidence_contradictions()
            
            return {
                "contradictions": {
                    "timeline": timeline_contradictions,
                    "alibi": alibi_contradictions,
                    "motive": motive_contradictions,
                    "evidence": evidence_contradictions
                },
                "trick_elements": self._identify_trick_elements(),
                "red_herrings": self._identify_red_herrings()
            }
            
        except Exception as e:
            print(f"[AnalysisAgent] 分析陷阱和矛盾時出錯: {str(e)}")
            return {"contradictions": [], "trick_elements": [], "red_herrings": []}
    
    def _analyze_timeline_contradictions(self) -> List[Dict[str, Any]]:
        """分析時間線矛盾"""
        try:
            # 查詢時間線信息
            timeline_data = self.nebula_service.execute_query(
                """
                MATCH (t:Timeline)-[:HAPPENED_ON]->(e:Event)
                MATCH (p:Person)-[:PARTICIPATED_IN]->(e)
                RETURN t.name as timeline, t.start as start_time, t.end as end_time,
                       e.name as event, p.name as participant
                ORDER BY t.start
                """
            )
            
            contradictions = []
            # 這裡可以實現更複雜的時間線分析邏輯
            # 簡化版本：檢查是否有重疊時間
            for i, event1 in enumerate(timeline_data):
                for j, event2 in enumerate(timeline_data[i+1:], i+1):
                    if self._check_time_overlap(event1, event2):
                        contradictions.append({
                            "type": "time_overlap",
                            "event1": event1.get("event", ""),
                            "event2": event2.get("event", ""),
                            "participant": event1.get("participant", ""),
                            "severity": "medium"
                        })
            
            return contradictions
            
        except Exception as e:
            print(f"[AnalysisAgent] 分析時間線矛盾時出錯: {str(e)}")
            return []
    
    def _analyze_alibi_contradictions(self) -> List[Dict[str, Any]]:
        """分析不在場證明矛盾"""
        try:
            # 查詢不在場證明信息
            alibi_data = self.nebula_service.execute_query(
                """
                MATCH (p:Person)-[:PARTICIPATED_IN]->(e:Event)
                WHERE e.name CONTAINS "不在場證明" OR e.description CONTAINS "不在場證明"
                RETURN p.name as person, e.name as event, e.description as description
                """
            )
            
            contradictions = []
            # 分析不在場證明的邏輯一致性
            for alibi in alibi_data:
                person = alibi.get("person", "")
                description = alibi.get("description", "")
                
                # 簡單的邏輯檢查
                if "矛盾" in description or "不一致" in description:
                    contradictions.append({
                        "type": "alibi_contradiction",
                        "person": person,
                        "description": description,
                        "severity": "high"
                    })
            
            return contradictions
            
        except Exception as e:
            print(f"[AnalysisAgent] 分析不在場證明矛盾時出錯: {str(e)}")
            return []
    
    def _analyze_motive_contradictions(self) -> List[Dict[str, Any]]:
        """分析動機矛盾"""
        try:
            # 查詢動機相關信息
            motive_data = self.nebula_service.execute_query(
                """
                MATCH (p:Person)-[:PARTICIPATED_IN]->(e:Event)
                WHERE e.description CONTAINS "動機" OR e.description CONTAINS "原因"
                RETURN p.name as person, e.name as event, e.description as description
                """
            )
            
            contradictions = []
            # 分析動機的一致性
            person_motives = {}
            for motive in motive_data:
                person = motive.get("person", "")
                description = motive.get("description", "")
                
                if person not in person_motives:
                    person_motives[person] = []
                person_motives[person].append(description)
            
            # 檢查同一人的動機是否矛盾
            for person, motives in person_motives.items():
                if len(motives) > 1:
                    # 簡單檢查：如果動機描述差異很大，可能存在矛盾
                    if self._check_motive_contradiction(motives):
                        contradictions.append({
                            "type": "motive_contradiction",
                            "person": person,
                            "motives": motives,
                            "severity": "medium"
                        })
            
            return contradictions
            
        except Exception as e:
            print(f"[AnalysisAgent] 分析動機矛盾時出錯: {str(e)}")
            return []
    
    def _analyze_evidence_contradictions(self) -> List[Dict[str, Any]]:
        """分析證據矛盾"""
        try:
            # 查詢證據相關信息
            evidence_data = self.nebula_service.execute_query(
                """
                MATCH (i:Item)-[:HAPPENED_AT]->(e:Event)
                WHERE i.type IN ["證據", "線索", "凶器"]
                RETURN i.name as evidence, i.type as type, e.name as event, e.description as description
                """
            )
            
            contradictions = []
            # 分析證據的一致性
            evidence_locations = {}
            for evidence in evidence_data:
                evidence_name = evidence.get("evidence", "")
                event = evidence.get("event", "")
                
                if evidence_name not in evidence_locations:
                    evidence_locations[evidence_name] = []
                evidence_locations[evidence_name].append(event)
            
            # 檢查同一證據是否出現在多個地方
            for evidence, events in evidence_locations.items():
                if len(events) > 1:
                    contradictions.append({
                        "type": "evidence_contradiction",
                        "evidence": evidence,
                        "events": events,
                        "severity": "high"
                    })
            
            return contradictions
            
        except Exception as e:
            print(f"[AnalysisAgent] 分析證據矛盾時出錯: {str(e)}")
            return []
    
    def _identify_trick_elements(self) -> List[Dict[str, Any]]:
        """識別陷阱元素"""
        try:
            # 查詢可能的陷阱元素
            trick_elements = self.nebula_service.execute_query(
                """
                MATCH (e:Event)
                WHERE e.description CONTAINS "假象" OR e.description CONTAINS "誤導" 
                   OR e.description CONTAINS "陷阱" OR e.description CONTAINS "詭計"
                RETURN e.name as name, e.description as description
                """
            )
            
            tricks = []
            for element in trick_elements:
                tricks.append({
                    "name": element.get("name", ""),
                    "description": element.get("description", ""),
                    "type": "trick_element"
                })
            
            return tricks
            
        except Exception as e:
            print(f"[AnalysisAgent] 識別陷阱元素時出錯: {str(e)}")
            return []
    
    def _identify_red_herrings(self) -> List[Dict[str, Any]]:
        """識別紅鯡魚（誤導線索）"""
        try:
            # 查詢可能的誤導線索
            red_herrings = self.nebula_service.execute_query(
                """
                MATCH (i:Item)
                WHERE i.type IN ["線索", "證據"] AND 
                      (i.name CONTAINS "假" OR i.description CONTAINS "誤導")
                RETURN i.name as name, i.type as type, i.description as description
                """
            )
            
            herrings = []
            for herring in red_herrings:
                herrings.append({
                    "name": herring.get("name", ""),
                    "type": herring.get("type", ""),
                    "description": herring.get("description", ""),
                    "type": "red_herring"
                })
            
            return herrings
            
        except Exception as e:
            print(f"[AnalysisAgent] 識別紅鯡魚時出錯: {str(e)}")
            return []
    
    def _generate_compression_suggestions(self, graph_stats: Dict, core_elements: Dict, 
                                        trick_analysis: Dict) -> Dict[str, Any]:
        """生成壓縮建議"""
        try:
            # 基於分析結果生成壓縮建議
            suggestions = {
                "character_merge_suggestions": self._suggest_character_merges(core_elements),
                "event_simplification": self._suggest_event_simplification(core_elements),
                "must_keep_contradictions": self._identify_must_keep_contradictions(trick_analysis),
                "compression_priority": self._generate_compression_priority(core_elements, trick_analysis),
                "estimated_compression_ratio": self._estimate_compression_ratio(graph_stats, core_elements)
            }
            
            return suggestions
            
        except Exception as e:
            print(f"[AnalysisAgent] 生成壓縮建議時出錯: {str(e)}")
            return self._get_default_compression_suggestions()
    
    def _suggest_character_merges(self, core_elements: Dict) -> List[Dict[str, str]]:
        """建議角色合併"""
        characters = core_elements.get("characters", [])
        merge_suggestions = []
        
        # 找出非關鍵角色
        non_key_characters = [char for char in characters if not char.get("is_key", False)]
        
        # 建議合併相似角色的功能
        for char in non_key_characters:
            if char.get("importance_score", 0) < 0.5:
                merge_suggestions.append({
                    "character": char.get("name", ""),
                    "suggestion": "可以與其他次要角色合併或簡化",
                    "reason": f"重要性分數較低: {char.get('importance_score', 0)}"
                })
        
        return merge_suggestions
    
    def _suggest_event_simplification(self, core_elements: Dict) -> List[Dict[str, str]]:
        """建議事件簡化"""
        events = core_elements.get("events", [])
        simplification_suggestions = []
        
        # 找出可以簡化的事件
        for event in events:
            if not event.get("is_key", False) and event.get("participant_count", 0) < 2:
                simplification_suggestions.append({
                    "event": event.get("name", ""),
                    "suggestion": "可以簡化或合併到其他事件中",
                    "reason": f"參與者較少: {event.get('participant_count', 0)}"
                })
        
        return simplification_suggestions
    
    def _identify_must_keep_contradictions(self, trick_analysis: Dict) -> List[Dict[str, str]]:
        """識別必須保留的矛盾"""
        contradictions = trick_analysis.get("contradictions", {})
        must_keep = []
        
        # 高嚴重性的矛盾必須保留
        for category, items in contradictions.items():
            for item in items:
                if item.get("severity") == "high":
                    must_keep.append({
                        "type": category,
                        "description": item.get("description", ""),
                        "reason": "高嚴重性矛盾，對推理至關重要"
                    })
        
        return must_keep
    
    def _generate_compression_priority(self, core_elements: Dict, trick_analysis: Dict) -> List[str]:
        """生成壓縮優先級"""
        priority = []
        
        # 1. 必須保留：關鍵角色和事件
        key_characters = [char for char in core_elements.get("characters", []) if char.get("is_key", False)]
        key_events = [event for event in core_elements.get("events", []) if event.get("is_key", False)]
        
        priority.append("必須保留所有關鍵角色和事件")
        
        # 2. 高優先級：重要矛盾
        must_keep_contradictions = self._identify_must_keep_contradictions(trick_analysis)
        if must_keep_contradictions:
            priority.append("必須保留所有高嚴重性矛盾")
        
        # 3. 中優先級：次要角色和事件
        priority.append("可以簡化次要角色和事件")
        
        # 4. 低優先級：背景描述
        priority.append("可以大幅壓縮背景描述和環境描寫")
        
        return priority
    
    def _estimate_compression_ratio(self, graph_stats: Dict, core_elements: Dict) -> str:
        """估算壓縮比例"""
        try:
            total_nodes = graph_stats.get("nodes", {}).get("total", 0)
            key_characters = len([char for char in core_elements.get("characters", []) if char.get("is_key", False)])
            key_events = len([event for event in core_elements.get("events", []) if event.get("is_key", False)])
            
            # 基於關鍵元素比例估算壓縮比例
            if total_nodes > 0:
                key_ratio = (key_characters + key_events) / total_nodes
                if key_ratio > 0.7:
                    return "20-30%"
                elif key_ratio > 0.5:
                    return "30-40%"
                elif key_ratio > 0.3:
                    return "40-50%"
                else:
                    return "50-60%"
            else:
                return "30-50%"
                
        except Exception as e:
            print(f"[AnalysisAgent] 估算壓縮比例時出錯: {str(e)}")
            return "30-50%"
    
    def _calculate_graph_density(self, node_stats: Dict, edge_stats: Dict) -> float:
        """計算圖密度"""
        try:
            total_nodes = node_stats.get("total", 0)
            total_edges = edge_stats.get("total", 0)
            
            if total_nodes <= 1:
                return 0.0
            
            # 圖密度 = 實際邊數 / 最大可能邊數
            max_edges = total_nodes * (total_nodes - 1)
            return round(total_edges / max_edges, 3) if max_edges > 0 else 0.0
            
        except Exception as e:
            print(f"[AnalysisAgent] 計算圖密度時出錯: {str(e)}")
            return 0.0
    
    def _check_time_overlap(self, event1: Dict, event2: Dict) -> bool:
        """檢查時間重疊"""
        try:
            start1 = event1.get("start_time", "")
            end1 = event1.get("end_time", "")
            start2 = event2.get("start_time", "")
            end2 = event2.get("end_time", "")
            
            # 簡化的時間重疊檢查
            if start1 and start2 and end1 and end2:
                # 這裡可以實現更複雜的時間比較邏輯
                return start1 == start2 or end1 == end2
            return False
            
        except Exception as e:
            print(f"[AnalysisAgent] 檢查時間重疊時出錯: {str(e)}")
            return False
    
    def _check_motive_contradiction(self, motives: List[str]) -> bool:
        """檢查動機矛盾"""
        try:
            # 簡單的關鍵詞檢查
            positive_keywords = ["愛", "保護", "正義", "幫助"]
            negative_keywords = ["恨", "報復", "嫉妒", "貪婪"]
            
            has_positive = any(any(keyword in motive for keyword in positive_keywords) for motive in motives)
            has_negative = any(any(keyword in motive for keyword in negative_keywords) for motive in motives)
            
            return has_positive and has_negative
            
        except Exception as e:
            print(f"[AnalysisAgent] 檢查動機矛盾時出錯: {str(e)}")
            return False
    
    def _get_default_analysis_result(self) -> Dict[str, Any]:
        """獲取默認分析結果"""
        return {
            "graph_statistics": {"nodes": {"total": 0}, "edges": {"total": 0}},
            "core_elements": {"characters": [], "events": []},
            "trick_analysis": {"contradictions": []},
            "compression_suggestions": {
                "character_merge_suggestions": [],
                "event_simplification": [],
                "must_keep_contradictions": [],
                "compression_priority": [],
                "estimated_compression_ratio": "30-50%"
            },
            "timestamp": self._get_current_timestamp()
        }
    
    def _get_default_compression_suggestions(self) -> Dict[str, Any]:
        """獲取默認壓縮建議"""
        return {
            "character_merge_suggestions": [],
            "event_simplification": [],
            "must_keep_contradictions": [],
            "compression_priority": ["保持核心情節", "簡化次要角色", "壓縮背景描述"],
            "estimated_compression_ratio": "30-50%"
        }
    
    def _get_current_timestamp(self) -> str:
        """獲取當前時間戳"""
        from datetime import datetime
        return datetime.now().isoformat()


class CompressionStrategy:
    """壓縮策略類"""
    
    def __init__(self, analysis_result: Dict[str, Any]):
        self.analysis = analysis_result
    
    def get_high_priority_elements(self) -> List[str]:
        """獲取高優先級保留元素"""
        return []
    
    def get_merge_candidates(self) -> List[Dict[str, str]]:
        """獲取可合併的角色候選"""
        return []
    
    def get_compression_target_ratio(self) -> str:
        """獲取目標壓縮比例"""
        return "30-50%"