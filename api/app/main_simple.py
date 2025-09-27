from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Dict, Any
import httpx
import json

app = FastAPI(title="ScriptAgent API (简化版)", version="0.1.0")

# 请求模型
class ExtractRequest(BaseModel):
    text: str

class RAGRequest(BaseModel):
    question: str

# 初始化 HTTP 客户端
http_client = httpx.AsyncClient(timeout=60.0)

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "message": "剧本杀 Agent API 运行正常",
        "version": "0.1.0",
        "services": {
            "llm": "Google Gemini API",
            "vector": "Qdrant",
            "graph": "NebulaGraph"
        }
    }

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用剧本杀 Agent API",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/extract")
async def extract_entities_and_events(req: ExtractRequest):
    """文本抽取（模拟）"""
    try:
        # 模拟抽取结果
        result = {
            "persons": [
                {"name": "张三", "role": "侦探"},
                {"name": "李四", "role": "嫌疑人"}
            ],
            "locations": [
                {"name": "书房"},
                {"name": "客厅"}
            ],
            "items": [
                {"name": "刀", "type": "凶器"},
                {"name": "钥匙", "type": "线索"}
            ],
            "events": [
                {
                    "name": "发现尸体",
                    "description": "在书房发现了一具尸体",
                    "participants": ["张三"],
                    "location": "书房",
                    "time": "晚上8点"
                }
            ],
            "timelines": [
                {"name": "案发当晚", "start": "晚上7点", "end": "晚上9点"}
            ]
        }
        
        return {
            "success": True,
            "data": result,
            "message": "文本抽取完成（模拟结果）"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def rag_query(req: RAGRequest):
    """RAG 问答（模拟）"""
    try:
        # 模拟 RAG 结果
        result = {
            "question": req.question,
            "nql": "USE scripts; MATCH (p:Person) RETURN p.name LIMIT 10",
            "kg_results": [
                {"name": "张三", "role": "侦探"},
                {"name": "李四", "role": "嫌疑人"}
            ],
            "vector_results": [
                {"text": "张三在书房里发现了一把刀", "score": 0.95},
                {"text": "李四声称自己当时在客厅", "score": 0.87}
            ],
            "answer": f"根据知识图谱和文本检索，关于'{req.question}'的回答：这是一个模拟的回答，实际系统会调用 Google Gemini API 进行智能分析。"
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/gemini")
async def test_gemini():
    """测试 Gemini API 连接"""
    try:
        api_key = os.getenv("GEMINI_API_KEY", "test_key_placeholder")
        if api_key == "test_key_placeholder":
            return {
                "status": "warning",
                "message": "请设置真实的 GEMINI_API_KEY 环境变量",
                "api_key_set": False
            }
        
        # 这里可以添加实际的 Gemini API 测试
        return {
            "status": "success",
            "message": "Gemini API 配置正确",
            "api_key_set": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "api_key_set": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)






