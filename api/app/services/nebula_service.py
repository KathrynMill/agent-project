import os
from typing import List, Dict, Any
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.gclient.net import Connection
from nebula3.gclient.graph import GraphSession
import json


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
        self.pool = ConnectionPool()
        addr_list = [(addr.split(":")[0], int(addr.split(":")[1])) for addr in self.addrs]
        ok = self.pool.init(addr_list, self.config)
        if not ok:
            raise RuntimeError("Failed to init Nebula connection pool")
    
    def get_session(self) -> GraphSession:
        """获取会话"""
        return self.pool.get_session(self.user, self.password)
    
    def execute_query(self, nql: str) -> Dict[str, Any]:
        """执行 nGQL 查询"""
        session = self.get_session()
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
