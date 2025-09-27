import os
from typing import List, Dict, Any
import json

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def post(self, *args, **kwargs):
            return MockResponse()
    class MockResponse:
        def json(self):
            return {"embeddings": [{"values": [0.1 for _ in range(768)]}]}
    httpx = type('httpx', (), {'AsyncClient': AsyncClient})()

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass
        def create_collection(self, *args, **kwargs):
            return True
        def upsert(self, *args, **kwargs):
            return True
        def search(self, *args, **kwargs):
            return []
    class Distance:
        COSINE = "Cosine"
    class VectorParams:
        def __init__(self, *args, **kwargs):
            pass
    class PointStruct:
        def __init__(self, *args, **kwargs):
            pass


class VectorService:
    """向量服务，使用 Google Gemini 嵌入模型"""
    
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "mock_key")
        self.gemini_api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        
        if QDRANT_AVAILABLE:
            self.client = QdrantClient(url=self.qdrant_url)
        else:
            self.client = QdrantClient(url=self.qdrant_url)
            
        if HTTPX_AVAILABLE:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        else:
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
        self.collection_name = "script_documents"
        
        if not self.gemini_api_key or self.gemini_api_key == "mock_key":
            print("Warning: Using mock vector service")
        
        self._init_collection()
    
    def _init_collection(self):
        """初始化向量集合"""
        try:
            # 检查集合是否存在
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # 创建新集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
                print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Error initializing collection: {e}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """使用 Google Gemini 获取文本的向量嵌入"""
        try:
            url = f"{self.gemini_api_base}/models/embedding-001:embedContent?key={self.gemini_api_key}"
            response = await self.http_client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "content": {"parts": [{"text": text}]}
                }
            )
            response.raise_for_status()
            result = response.json()
            
            if "embedding" in result:
                return result["embedding"]["values"]
            else:
                raise Exception("Gemini 嵌入 API 返回格式错误")
                
        except Exception as e:
            raise Exception(f"获取嵌入向量失败: {str(e)}")
    
    async def add_document(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """添加文档到向量库"""
        try:
            # 获取嵌入向量
            embedding = await self.get_embedding(text)
            
            # 生成唯一 ID
            import uuid
            doc_id = str(uuid.uuid4())
            
            # 准备元数据
            if metadata is None:
                metadata = {}
            metadata["text"] = text
            
            # 添加到 Qdrant
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload=metadata
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    async def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        try:
            # 获取查询的嵌入向量
            query_embedding = await self.get_embedding(query)
            
            # 在 Qdrant 中搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # 格式化结果
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "score": result.score
                })
            
            return results
        except Exception as e:
            print(f"Error searching similar documents: {e}")
            return []
    
    async def batch_add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量添加文档"""
        results = {"success": True, "added": 0, "errors": []}
        
        for doc in documents:
            try:
                text = doc.get("text", "")
                metadata = {k: v for k, v in doc.items() if k != "text"}
                
                if await self.add_document(text, metadata):
                    results["added"] += 1
                else:
                    results["errors"].append(f"Failed to add document: {doc.get('title', 'Unknown')}")
            except Exception as e:
                results["errors"].append(f"Error processing document: {str(e)}")
        
        if results["errors"]:
            results["success"] = False
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
