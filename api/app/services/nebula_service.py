import os
from typing import List, Dict, Any
import json

try:
    from nebula3.gclient.net import ConnectionPool
    from nebula3.Config import Config
    from nebula3.gclient.net import Connection
    NEBULA_AVAILABLE = True
except ImportError:
    NEBULA_AVAILABLE = False
    # 創建模擬類
    class ConnectionPool:
        def __init__(self, *args, **kwargs):
            pass
    class Config:
        def __init__(self):
            self.minConnectionPoolSize = 1
            self.maxConnectionPoolSize = 10
    class Connection:
        def __init__(self, *args, **kwargs):
            pass


class NebulaService:
    def __init__(self):
        self.addrs = os.getenv("NEBULA_ADDRS", "nebula-graphd0:9669").split(",")
        self.user = os.getenv("NEBULA_USER", "root")
        self.password = os.getenv("NEBULA_PASSWORD", "")
        self.space = os.getenv("NEBULA_SPACE", "scripts")
        self.config = Config()
        self.config.minConnectionPoolSize = 1
        self.config.maxConnectionPoolSize = 10
        self.pool = None
        self._init_pool()
    
    def _init_pool(self):
        """初始化连接池"""
        if not NEBULA_AVAILABLE:
            print("Nebula3 not available, using mock service")
            self.pool = None
            return
            
        try:
            self.pool = ConnectionPool()
            addr_list = [(addr.split(":")[0], int(addr.split(":")[1])) for addr in self.addrs]
            ok = self.pool.init(addr_list, self.config)
            if not ok:
                raise RuntimeError("Failed to init Nebula connection pool")
        except Exception as e:
            print(f"Failed to initialize NebulaGraph connection pool: {e}")
            self.pool = None
    
    def get_session(self):
        """获取会话"""
        if not NEBULA_AVAILABLE or not self.pool:
            return None
        return self.pool.get_session(self.user, self.password)
    
    def execute_query(self, nql: str) -> Dict[str, Any]:
        """执行 nGQL 查询"""
        if not NEBULA_AVAILABLE or not self.pool:
            print("Nebula3 not available, returning mock result")
            return {"success": True, "data": [], "mock": True}
            
        session = self.get_session()
        if not session:
            return {"success": False, "error": "No session available", "mock": True}
            
        try:
            result = session.execute(nql)
            if not result.is_succeeded():
                return {"success": False, "error": str(result.error_msg())}
            
            # 转换结果为字典格式
            columns = result.keys()
            data = []
            for row in result.rows():
                row_data = {}
                for i, col in enumerate(columns):
                    row_data[col] = str(row.values()[i])
                data.append(row_data)
            
            return {"success": True, "columns": columns, "data": data}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            session.release()
    
    def upsert_person(self, name: str, role: str = "") -> bool:
        """插入或更新人物"""
        nql = f"""
        USE {self.space};
        INSERT VERTEX Person(name, role) VALUES "{name}":("{name}", "{role}");
        """
        result = self.execute_query(nql)
        return result["success"]
    
    def upsert_location(self, name: str) -> bool:
        """插入或更新地点"""
        nql = f"""
        USE {self.space};
        INSERT VERTEX Location(name) VALUES "{name}":("{name}");
        """
        result = self.execute_query(nql)
        return result["success"]
    
    def upsert_item(self, name: str, item_type: str = "") -> bool:
        """插入或更新物品"""
        nql = f"""
        USE {self.space};
        INSERT VERTEX Item(name, type) VALUES "{name}":("{name}", "{item_type}");
        """
        result = self.execute_query(nql)
        return result["success"]
    
    def upsert_event(self, name: str, description: str = "") -> bool:
        """插入或更新事件"""
        nql = f"""
        USE {self.space};
        INSERT VERTEX Event(name, description) VALUES "{name}":("{name}", "{description}");
        """
        result = self.execute_query(nql)
        return result["success"]
    
    def upsert_timeline(self, name: str, start: str = "", end: str = "") -> bool:
        """插入或更新时间线"""
        nql = f"""
        USE {self.space};
        INSERT VERTEX Timeline(name, start, end) VALUES "{name}":("{name}", "{start}", "{end}");
        """
        result = self.execute_query(nql)
        return result["success"]
    
    def create_relationship(self, from_vertex: str, to_vertex: str, edge_type: str, properties: Dict[str, str] = None) -> bool:
        """创建关系"""
        prop_str = ""
        if properties:
            prop_pairs = [f"{k}: \"{v}\"" for k, v in properties.items()]
            prop_str = f"({', '.join(prop_pairs)})"
        
        nql = f"""
        USE {self.space};
        INSERT EDGE {edge_type}{prop_str} VALUES "{from_vertex}" -> "{to_vertex}";
        """
        result = self.execute_query(nql)
        return result["success"]
    
    def batch_upsert_from_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """批量从 JSON 数据插入到图谱"""
        results = {"success": True, "errors": []}
        
        try:
            # 插入人物
            for person in data.get("persons", []):
                if not self.upsert_person(person["name"], person.get("role", "")):
                    results["errors"].append(f"Failed to upsert person: {person['name']}")
            
            # 插入地点
            for location in data.get("locations", []):
                if not self.upsert_location(location["name"]):
                    results["errors"].append(f"Failed to upsert location: {location['name']}")
            
            # 插入物品
            for item in data.get("items", []):
                if not self.upsert_item(item["name"], item.get("type", "")):
                    results["errors"].append(f"Failed to upsert item: {item['name']}")
            
            # 插入事件
            for event in data.get("events", []):
                if not self.upsert_event(event["name"], event.get("description", "")):
                    results["errors"].append(f"Failed to upsert event: {event['name']}")
            
            # 插入时间线
            for timeline in data.get("timelines", []):
                if not self.upsert_timeline(timeline["name"], timeline.get("start", ""), timeline.get("end", "")):
                    results["errors"].append(f"Failed to upsert timeline: {timeline['name']}")
            
            # 创建关系（这里需要根据具体的剧本逻辑来定义关系）
            # 示例：人物参与事件
            for event in data.get("events", []):
                event_name = event["name"]
                for participant in event.get("participants", []):
                    if not self.create_relationship(participant, event_name, "PARTICIPATED_IN"):
                        results["errors"].append(f"Failed to create relationship: {participant} -> {event_name}")
                
                # 事件发生于地点
                if event.get("location"):
                    if not self.create_relationship(event_name, event["location"], "HAPPENED_AT"):
                        results["errors"].append(f"Failed to create relationship: {event_name} -> {event['location']}")
            
            if results["errors"]:
                results["success"] = False
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
        
        return results

    def batch_upsert_from_graph_data(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """批量入库：支持 trick-aware 三层结构（truths/statements/tricks）。
        期望输入格式：
        {
          "persons": [...], "locations": [...], "items": [...], "events": [...], "timelines": [...],
          "statements": [{"id":"s1","content":"...","perspective":"角色A","source_chunk_id":"c1"}, ...],
          "truths": [{"event":"老爷死亡","real_time":"7:50"}],
          "tricks": [{"type":"CONTRADICTS","from":"s2","to":"s1"}, {"type":"POTENTIAL_ALIAS_OF","from":"角色A","to":"角色B"}]
        }
        """
        if not NEBULA_AVAILABLE or not self.pool:
            print("Nebula3 not available, returning mock result")
            return {"success": True, "errors": [], "mock": True}
            
        results = {"success": True, "errors": []}
        try:
            # 先复用既有入库（实体/事件/时间线）
            base = {
                "persons": graph.get("persons", []),
                "locations": graph.get("locations", []),
                "items": graph.get("items", []),
                "events": graph.get("events", []),
                "timelines": graph.get("timelines", [])
            }
            base_result = self.batch_upsert_from_json(base)
            if not base_result["success"]:
                results["errors"].extend(base_result["errors"])

            # 插入 Statement 节点
            for st in graph.get("statements", []):
                sid = st.get("id") or st.get("content")
                content = st.get("content", "")
                perspective = st.get("perspective", "")
                chunk_id = st.get("source_chunk_id", "")
                nql = f"""
                USE {self.space};
                INSERT VERTEX Statement(content, perspective, source_chunk_id)
                VALUES "{sid}":("{content}", "{perspective}", "{chunk_id}");
                """
                r = self.execute_query(nql)
                if not r["success"]:
                    results["errors"].append(f"Failed to upsert statement: {sid} -> {r['error']}")

            # 插入诡计关系
            for tr in graph.get("tricks", []):
                t = tr.get("type")
                src = tr.get("from")
                dst = tr.get("to")
                if not t or not src or not dst:
                    results["errors"].append(f"Invalid trick edge: {tr}")
                    continue
                if t not in ("CONTRADICTS", "POTENTIAL_ALIAS_OF"):
                    results["errors"].append(f"Unsupported trick type: {t}")
                    continue
                nql = f"""
                USE {self.space};
                INSERT EDGE {t}() VALUES "{src}" -> "{dst}";
                """
                r = self.execute_query(nql)
                if not r["success"]:
                    results["errors"].append(f"Failed to create trick edge {t}: {src}->{dst} -> {r['error']}")

            if results["errors"]:
                results["success"] = False
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results
    
    def search_entities(self, entity_type: str, name_pattern: str = "") -> List[Dict]:
        """搜索实体"""
        if name_pattern:
            nql = f"""
            USE {self.space};
            MATCH (n:{entity_type}) WHERE n.name CONTAINS "{name_pattern}" RETURN n;
            """
        else:
            nql = f"""
            USE {self.space};
            MATCH (n:{entity_type}) RETURN n;
            """
        
        result = self.execute_query(nql)
        if result["success"]:
            return result["data"]
        return []
    
    def get_relationships(self, entity_name: str, relationship_type: str = "") -> List[Dict]:
        """获取实体的关系"""
        if relationship_type:
            nql = f"""
            USE {self.space};
            MATCH (n)-[r:{relationship_type}]->(m) WHERE n.name == "{entity_name}" RETURN n, r, m;
            """
        else:
            nql = f"""
            USE {self.space};
            MATCH (n)-[r]->(m) WHERE n.name == "{entity_name}" RETURN n, r, m;
            """
        
        result = self.execute_query(nql)
        if result["success"]:
            return result["data"]
        return []
