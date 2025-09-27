from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Dict, Any
from app.services.llm_service import LLMService
from app.services.nebula_service import NebulaService
from app.services.vector_service import VectorService
from app.agents.chief_editor_agent import ChiefEditorAgent


app = FastAPI(title="ScriptAgent API", version="0.1.0")

# 初始化服务
llm_service = LLMService()
nebula_service = NebulaService()
vector_service = VectorService()
chief_editor = ChiefEditorAgent(nebula_service, llm_service)


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

class CompressRequest(BaseModel):
    player_scripts: Dict[str, str]
    master_guide: str
    target_hours: int = 4


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


@app.post("/compress_script")
async def compress_script(req: CompressRequest):
    """核心工作流：構建詭計感知圖譜 -> 多智能體壓縮 -> 返回最終稿"""
    try:
        # 1) 構建詭計感知圖譜
        print("開始構建詭計感知圖譜...")
        graph_data = llm_service.build_trick_aware_kg(req.player_scripts, req.master_guide)
        upsert_result = nebula_service.batch_upsert_from_graph_data(graph_data)
        print("圖譜構建完成")

        # 2) 創建並執行多智能體壓縮工作流
        print("開始多智能體壓縮工作流...")
        workflow = chief_editor.create_compression_workflow()
        
        # 準備工作流輸入數據
        workflow_input = {
            "player_scripts": req.player_scripts,
            "master_guide": req.master_guide,
            "target_hours": req.target_hours
        }
        
        # 執行工作流
        compression_result = workflow.invoke(workflow_input)
        print("多智能體壓縮工作流完成")

        return {
            "kg_upsert": upsert_result,
            "compression_result": compression_result,
            "message": "劇本壓縮完成"
        }
    except Exception as e:
        print(f"壓縮過程中發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



