"""图数据库服务 - NebulaGraph操作封装"""

import logging
from typing import Dict, List, Any, Optional
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import *
from nebula3.data.DataObject import DataObject
from nebula3.data.ResultSet import ResultSet

from ...shared.config.settings import get_settings
from ...shared.utils.exceptions import NebulaGraphError

logger = logging.getLogger(__name__)
settings = get_settings()


class NebulaGraphService:
    """NebulaGraph 图数据库服务"""

    def __init__(self):
        """初始化图数据库服务"""
        self.settings = settings.database
        self.connection_pool = None
        self.session = None
        self._initialized = False

    async def initialize(self):
        """初始化连接"""
        try:
            # 创建连接池
            config = Config()
            config.max_connection_pool_size = 10
            self.connection_pool = ConnectionPool()

            # 连接到 NebulaGraph
            if not self.connection_pool.init([
                (self.settings.nebula_host, self.settings.nebula_port)
            ], config):
                raise NebulaGraphError("无法连接到 NebulaGraph")

            # 获取会话
            self.session = self.connection_pool.get_session(
                self.settings.nebula_username,
                self.settings.nebula_password
            )

            self._initialized = True
            logger.info("NebulaGraph 连接初始化成功")

        except Exception as e:
            logger.error(f"NebulaGraph 初始化失败: {str(e)}")
            raise NebulaGraphError(f"初始化失败: {str(e)}")

    async def execute_query(self, query: str) -> ResultSet:
        """
        执行 nGQL 查询

        Args:
            query: nGQL 查询语句

        Returns:
            ResultSet: 查询结果
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.debug(f"执行 nGQL: {query}")
            result = self.session.execute(query)

            if not result.is_succeeded():
                error_msg = result.error_msg()
                raise NebulaGraphError(f"查询失败: {error_msg}")

            return result

        except Exception as e:
            logger.error(f"nGQL 查询失败: {str(e)}")
            raise NebulaGraphError(f"查询执行失败: {str(e)}")

    async def ping(self) -> bool:
        """
        检查连接状态

        Returns:
            bool: 连接是否正常
        """
        try:
            result = await self.execute_query("YIELD 1")
            return result.is_succeeded()
        except Exception as e:
            logger.error(f"NebulaGraph ping 失败: {str(e)}")
            return False

    async def close(self):
        """关闭连接"""
        try:
            if self.session:
                self.session.release()
            if self.connection_pool:
                self.connection_pool.close()
            self._initialized = False
            logger.info("NebulaGraph 连接已关闭")
        except Exception as e:
            logger.error(f"关闭 NebulaGraph 连接失败: {str(e)}")

    def result_to_dict(self, result: ResultSet) -> List[Dict[str, Any]]:
        """
        将 ResultSet 转换为字典列表

        Args:
            result: ResultSet 对象

        Returns:
            List[Dict[str, Any]]: 结果数据
        """
        if not result.is_succeeded():
            return []

        data = []
        columns = [col for col in result.col_names()]

        for record in result:
            row = {}
            for i, value in enumerate(record.values):
                if isinstance(value, DataObject):
                    # 根据数据类型转换
                    if value.is_int():
                        row[columns[i]] = value.as_int()
                    elif value.is_bool():
                        row[columns[i]] = value.as_bool()
                    elif value.is_double():
                        row[columns[i]] = value.as_double()
                    elif value.is_string():
                        row[columns[i]] = value.as_string()
                    elif value.is_vertex():
                        vertex = value.as_node()
                        row[columns[i]] = {
                            "id": vertex.get_id().as_string(),
                            "tags": vertex.tags()
                        }
                    elif value.is_edge():
                        edge = value.as_relationship()
                        row[columns[i]] = {
                            "src": edge.src().as_string(),
                            "dst": edge.dst().as_string(),
                            "type": edge.name()
                        }
                    else:
                        row[columns[i]] = str(value)
                else:
                    row[columns[i]] = str(value)

            data.append(row)

        return data