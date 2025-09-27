import os
from typing import Dict, Any, List
import json

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    # 創建模擬 httpx 類
    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def post(self, *args, **kwargs):
            return MockResponse()
    class MockResponse:
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Mock response"}]}}]}
    httpx = type('httpx', (), {'AsyncClient': AsyncClient})()


class LLMService:
    """LLM 服务，使用 Google Gemini API"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "mock_key")
        self.api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        
        if HTTPX_AVAILABLE:
            self.client = httpx.AsyncClient(timeout=60.0)
        else:
            self.client = httpx.AsyncClient(timeout=60.0)
        
        if not self.api_key or self.api_key == "mock_key":
            print("Warning: Using mock LLM service")
    
    async def generate_text(self, prompt: str, max_tokens: int = 2000) -> str:
        """使用 Google Gemini API 生成文本"""
        if not HTTPX_AVAILABLE or self.api_key == "mock_key":
            return f"Mock response for: {prompt[:100]}..."
            
        try:
            url = f"{self.api_base}/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            response = await self.client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": 0.7
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raise Exception("Gemini API 返回空结果")
                
        except Exception as e:
            raise Exception(f"Gemini API 调用失败: {str(e)}")
    
    async def extract_entities_and_events(self, text: str) -> Dict[str, Any]:
        """从剧本文本中抽取实体和事件"""
        prompt = f"""
请从以下剧本文本中抽取实体和事件，并以 JSON 格式返回。

抽取规则：
1. 人物 (Person): 包含姓名、角色描述
2. 地点 (Location): 具体地点名称
3. 物品 (Item): 重要物品，包含类型
4. 事件 (Event): 包含事件名称、描述、参与人物、发生地点、时间
5. 时间线 (Timeline): 时间点或时间段

返回格式：
{{
    "persons": [{{"name": "姓名", "role": "角色描述"}}],
    "locations": [{{"name": "地点名称"}}],
    "items": [{{"name": "物品名称", "type": "物品类型"}}],
    "events": [{{"name": "事件名称", "description": "事件描述", "participants": ["人物1", "人物2"], "location": "地点", "time": "时间"}}],
    "timelines": [{{"name": "时间名称", "start": "开始时间", "end": "结束时间"}}]
}}

剧本文本：
{text}
"""
        
        result = await self.generate_text(prompt, max_tokens=3000)
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # 如果 JSON 解析失败，尝试清理并重新解析
            cleaned = result.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned)
    
    async def text_to_ngql(self, question: str) -> str:
        """将自然语言问题转换为 nGQL 查询"""
        prompt = f"""
你是一个 NebulaGraph 专家。请根据用户的问题，生成对应的 nGQL 查询语句。

可用的标签 (Tags):
- Person: 人物 (属性: name, role)
- Location: 地点 (属性: name)
- Item: 物品 (属性: name, type)
- Event: 事件 (属性: name, description)
- Timeline: 时间线 (属性: name, start, end)

可用的边 (Edges):
- LOCATED_IN: 位于 (属性: time)
- HAS_ITEM: 拥有物品
- PARTICIPATED_IN: 参与事件 (属性: role)
- HAPPENED_AT: 发生于地点
- HAPPENED_ON: 发生于时间

查询规则：
1. 使用 USE scripts; 切换到正确的空间
2. 使用 MATCH 语句进行查询
3. 使用 RETURN 返回需要的结果
4. 如果需要过滤，使用 WHERE 子句

用户问题：{question}

请只返回 nGQL 查询语句，不要包含其他解释：
"""
        
        result = await self.generate_text(prompt, max_tokens=1000)
        # 清理结果，确保只返回 nGQL 语句
        lines = result.strip().split('\n')
        ngql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                ngql_lines.append(line)
        
        return '\n'.join(ngql_lines)
    
    async def generate_answer(self, question: str, kg_results: List[Dict], vector_results: List[str] = None) -> str:
        """基于知识图谱和向量检索结果生成答案"""
        kg_context = ""
        if kg_results:
            kg_context = "知识图谱检索结果：\n"
            for i, result in enumerate(kg_results, 1):
                kg_context += f"{i}. {result}\n"
        
        vector_context = ""
        if vector_results:
            vector_context = "\n相关文本片段：\n"
            for i, text in enumerate(vector_results, 1):
                vector_context += f"{i}. {text}\n"
        
        prompt = f"""
请基于以下信息回答用户的问题。如果信息不足，请明确说明。

{kg_context}
{vector_context}

用户问题：{question}

请提供准确、有根据的回答，并在可能的情况下引用具体的证据：
"""
        
        return await self.generate_text(prompt, max_tokens=1500)

    async def build_trick_aware_kg(self, player_scripts: Dict[str, str], master_guide: str) -> Dict[str, Any]:
        """构建詭計感知圖譜：
        1) 先讀主持人手冊抽取真相層 truths
        2) 再讀玩家劇本抽取各自陳述 statements，並對比真相生成 tricks（如 CONTRADICTS）
        返回：包含 persons/locations/items/events/timelines/statements/truths/tricks 的字典
        """
        # 第一階段：真相層
        truth_prompt = f"""
你是案件整理專家。以下是主持人手冊（真相層）。
請抽取：
1) 事件真相（關鍵事件與真實時間 real_time）
2) 真實角色關係（人-人、人-事件、人-地點）
3) 重要時間線
請輸出 JSON，鍵包含：events（含 real_time）、persons、locations、items、timelines、truths。

主持人手冊：\n{master_guide}
"""
        truths_json = await self.generate_text(truth_prompt, max_tokens=3000)
        try:
            truths = json.loads(truths_json)
        except Exception:
            truths = {"events": [], "persons": [], "locations": [], "items": [], "timelines": [], "truths": []}

        # 第二階段：玩家陳述層 + 詭計
        statements: List[Dict[str, Any]] = []
        tricks: List[Dict[str, str]] = []
        for role, script in player_scripts.items():
            st_prompt = f"""
你是推理抽取器。針對以下玩家視角文本，抽取該角色的關鍵陳述（Statement）。
每條陳述需包含：id（自定義唯一）、content（內容）、perspective（角色名）、source_chunk_id（可留空）。
同時，對比以下真相層 truths，給出矛盾關係 CONTRADICTS（from 為此陳述 id、to 為相矛盾的真相事件或陳述 id 名稱）。
輸出 JSON：{{"statements": [...], "tricks": [...]}}

玩家角色：{role}
玩家文本：\n{script}
真相層（供對比）：\n{json.dumps(truths, ensure_ascii=False)}
"""
            st_json = await self.generate_text(st_prompt, max_tokens=3000)
            try:
                part = json.loads(st_json)
                statements.extend(part.get("statements", []))
                tricks.extend(part.get("tricks", []))
            except Exception:
                continue

        merged = {
            "persons": truths.get("persons", []),
            "locations": truths.get("locations", []),
            "items": truths.get("items", []),
            "events": truths.get("events", []),
            "timelines": truths.get("timelines", []),
            "truths": truths.get("truths", []),
            "statements": statements,
            "tricks": tricks,
        }
        return merged
