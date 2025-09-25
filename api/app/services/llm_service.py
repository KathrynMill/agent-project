import os
import httpx
from typing import Dict, Any, List
import json


class LLMService:
    def __init__(self):
        self.api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
        self.api_key = os.getenv("OPENAI_API_KEY", "local")
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate_text(self, prompt: str, max_tokens: int = 2000) -> str:
        """调用 LLM 生成文本"""
        try:
            response = await self.client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Qwen/Qwen2.5-14B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"LLM 调用失败: {str(e)}")
    
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
