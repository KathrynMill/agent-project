from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Dict, Any
from services.llm_service import LLMService
from services.nebula_service import NebulaService
from services.vector_service import VectorService


app = FastAPI(title="ScriptAgent API", version="0.1.0")

# 初始化服务
llm_service = LLMService()
nebula_service = NebulaService()
vector_service = VectorService()


class Health(BaseModel):
    status: str


class QueryRequest(BaseModel):
    nql: str


class TextToNGQLRequest(BaseModel):
    question: str


class ExtractRequest(BaseModel):
    text: str


class RAGRequest(BaseModel):
    question: str


@app.get("/health", response_model=Health)
async def health() -> Health:
    return Health(status="ok")


@app.get("/nebula/ping")
async def nebula_ping():
    try:
        result = nebula_service.execute_query("SHOW SPACES;")
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"spaces": result["data"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nebula/query")
async def nebula_query(req: QueryRequest):
    result = nebula_service.execute_query(req.nql)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"columns": result["columns"], "data": result["data"]}


@app.post("/text-to-ngql")
async def text_to_ngql(req: TextToNGQLRequest):
    try:
        nql = await llm_service.text_to_ngql(req.question)
        return {"nql": nql}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract_entities(req: ExtractRequest):
    try:
        # 抽取实体和事件
        extracted_data = await llm_service.extract_entities_and_events(req.text)
        
        # 批量插入到图谱
        upsert_result = nebula_service.batch_upsert_from_json(extracted_data)
        
        # 同时将文本添加到向量库
        vector_result = await vector_service.add_document(req.text, {"type": "script_text"})
        
        return {
            "extracted": extracted_data,
            "kg_upsert": upsert_result,
            "vector_added": vector_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query")
async def rag_query(req: RAGRequest):
    try:
        # 1. 将问题转换为 nGQL
        nql = await llm_service.text_to_ngql(req.question)
        
        # 2. 执行图谱查询
        kg_result = nebula_service.execute_query(nql)
        kg_data = kg_result.get("data", []) if kg_result["success"] else []
        
        # 3. 向量检索
        vector_results = await vector_service.search_similar(req.question, limit=3)
        vector_texts = [r["text"] for r in vector_results]
        
        # 4. 生成最终答案
        answer = await llm_service.generate_answer(req.question, kg_data, vector_texts)
        
        return {
            "question": req.question,
            "nql": nql,
            "kg_results": kg_data,
            "vector_results": vector_results,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector/info")
async def vector_info():
    return vector_service.get_collection_info()


@app.post("/vector/search")
async def vector_search(query: str, limit: int = 5):
    results = await vector_service.search_similar(query, limit)
    return {"query": query, "results": results}



