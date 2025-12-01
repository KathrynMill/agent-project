"""向量数据库服务 - Qdrant操作封装"""

import logging
from typing import Dict, List, Any, Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import numpy as np

from ...shared.config.settings import get_settings
from ...shared.utils.exceptions import QdrantError

logger = logging.getLogger(__name__)
settings = get_settings()


class QdrantService:
    """Qdrant 向量数据库服务"""

    def __init__(self):
        """初始化向量数据库服务"""
        self.settings = settings.database
        self.client = None
        self._initialized = False

    async def initialize(self):
        """初始化连接"""
        try:
            # 创建 Qdrant 客户端
            self.client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                timeout=30
            )

            # 测试连接
            collections = self.client.get_collections()
            logger.info(f"Qdrant 连接成功，现有集合: {[c.name for c in collections.collections]}")

            # 确保目标集合存在
            await self._ensure_collection_exists()

            self._initialized = True
            logger.info("Qdrant 初始化成功")

        except Exception as e:
            logger.error(f"Qdrant 初始化失败: {str(e)}")
            raise QdrantError(f"初始化失败: {str(e)}")

    async def _ensure_collection_exists(self):
        """确保集合存在"""
        try:
            collection_name = self.settings.qdrant_collection_name
            collections = self.client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if collection_name not in existing_names:
                # 创建集合
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建向量集合: {collection_name}")
            else:
                logger.info(f"向量集合已存在: {collection_name}")

        except Exception as e:
            logger.error(f"确保集合存在失败: {str(e)}")
            raise QdrantError(f"集合操作失败: {str(e)}")

    async def insert_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        插入向量数据

        Args:
            vectors: 向量列表
            payloads: 对应的元数据
            ids: 可选的ID列表

        Returns:
            List[str]: 插入的点ID列表
        """
        if not self._initialized:
            await self.initialize()

        try:
            collection_name = self.settings.qdrant_collection_name

            # 如果没有提供ID，生成UUID
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            # 创建点结构
            points = []
            for i, (vector, payload, point_id) in enumerate(zip(vectors, payloads, ids)):
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))

            # 批量插入
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info(f"成功插入 {len(points)} 个向量")
            return ids

        except Exception as e:
            logger.error(f"插入向量失败: {str(e)}")
            raise QdrantError(f"插入向量失败: {str(e)}")

    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_condition: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量

        Args:
            query_vector: 查询向量
            limit: 返回结果数量
            score_threshold: 相似度阈值
            filter_condition: 过滤条件

        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        if not self._initialized:
            await self.initialize()

        try:
            collection_name = self.settings.qdrant_collection_name

            # 执行搜索
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_condition
            )

            # 格式化结果
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })

            logger.info(f"搜索到 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            raise QdrantError(f"向量搜索失败: {str(e)}")

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息

        Returns:
            Dict[str, Any]: 集合信息
        """
        if not self._initialized:
            await self.initialize()

        try:
            collection_name = self.settings.qdrant_collection_name
            info = self.client.get_collection(collection_name)

            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "vector_size": info.config.params.vectors.size
            }

        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            raise QdrantError(f"获取集合信息失败: {str(e)}")

    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        删除向量

        Args:
            ids: 要删除的点ID列表

        Returns:
            bool: 是否成功
        """
        if not self._initialized:
            await self.initialize()

        try:
            collection_name = self.settings.qdrant_collection_name
            self.client.delete(
                collection_name=collection_name,
                points_selector=ids
            )
            logger.info(f"成功删除 {len(ids)} 个向量")
            return True

        except Exception as e:
            logger.error(f"删除向量失败: {str(e)}")
            raise QdrantError(f"删除向量失败: {str(e)}")

    async def ping(self) -> bool:
        """
        检查连接状态

        Returns:
            bool: 连接是否正常
        """
        try:
            if not self._initialized:
                await self.initialize()

            # 尝试获取集合列表来测试连接
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant ping 失败: {str(e)}")
            return False

    async def close(self):
        """关闭连接"""
        try:
            if self.client:
                self.client.close()
            self._initialized = False
            logger.info("Qdrant 连接已关闭")
        except Exception as e:
            logger.error(f"关闭 Qdrant 连接失败: {str(e)}")