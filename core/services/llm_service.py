"""大语言模型服务 - 提供LLM调用能力"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import httpx
from pydantic import BaseModel

from ...shared.config.settings import get_settings
from ...shared.utils.exceptions import LLMServiceError, ModelInferenceError

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMRequest(BaseModel):
    """LLM请求模型"""
    prompt: str
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None


class LLMResponse(BaseModel):
    """LLM响应模型"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


class LLMService:
    """大语言模型服务"""

    def __init__(self):
        """初始化LLM服务"""
        self.settings = settings
        self.base_url = settings.get_llm_base_url()
        self.api_key = settings.llm.gemini_api_key
        self.model = settings.llm.gemini_model
        self.temperature = settings.llm.gemini_temperature

        # HTTP客户端配置
        self.client_config = {
            "timeout": httpx.Timeout(60.0),
            "limits": httpx.Limits(max_keepalive_connections=5, max_connections=10)
        }

        logger.info(f"LLM服务初始化完成，模型: {self.model}")

    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """
        生成文本

        Args:
            request: LLM请求

        Returns:
            LLMResponse: 生成的文本响应
        """
        try:
            # 构建请求体
            messages = []

            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})

            messages.append({"role": "user", "content": request.prompt})

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature or self.temperature,
                "top_p": request.top_p,
                "stream": False
            }

            async with httpx.AsyncClient(**self.client_config) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                )
                response.raise_for_status()

                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]

                return LLMResponse(
                    content=message["content"],
                    model=data["model"],
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason", "stop")
                )

        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API HTTP错误: {e.response.status_code} - {e.response.text}")
            raise LLMServiceError(f"LLM API请求失败: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"LLM API请求错误: {str(e)}")
            raise LLMServiceError(f"无法连接到LLM服务: {str(e)}")
        except Exception as e:
            logger.error(f"LLM文本生成失败: {str(e)}")
            raise ModelInferenceError(f"文本生成失败: {str(e)}")

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        从文本中提取实体

        Args:
            text: 输入文本

        Returns:
            Dict[str, List[str]]: 提取的实体字典
        """
        prompt = f"""
        请从以下剧本文本中提取关键实体，并按类别组织：

        文本：
        {text}

        请提取以下类型的实体：
        1. 人物(Person): 角色名称
        2. 地点(Location): 场景位置
        3. 物品(Item): 重要物品
        4. 事件(Event): 重要事件
        5. 时间(Time): 时间信息

        请以JSON格式返回结果，例如：
        {{
            "persons": ["张三", "李四"],
            "locations": ["客厅", "厨房"],
            "items": ["刀", "钥匙"],
            "events": ["谋杀", "发现"],
            "times": ["晚上8点", "午夜"]
        }}
        """

        request = LLMRequest(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.3  # 降低温度以获得更一致的结果
        )

        try:
            response = await self.generate_text(request)

            # 尝试解析JSON响应
            import json
            try:
                entities = json.loads(response.content)
                return entities
            except json.JSONDecodeError:
                logger.warning("LLM返回的不是有效JSON，尝试解析文本")
                # 简单的文本解析逻辑
                return self._parse_entities_from_text(response.content)

        except Exception as e:
            logger.error(f"实体提取失败: {str(e)}")
            raise LLMServiceError(f"实体提取失败: {str(e)}")

    async def extract_relations(self, entities: Dict[str, List[str]], text: str) -> List[Dict[str, Any]]:
        """
        提取实体之间的关系

        Args:
            entities: 实体字典
            text: 输入文本

        Returns:
            List[Dict[str, Any]]: 关系列表
        """
        prompt = f"""
        基于以下实体和文本，提取实体之间的关系：

        实体：
        {entities}

        文本：
        {text}

        请识别以下类型的关系：
        1. LOCATED_IN: 人物在某个地点
        2. HAS_ITEM: 人物拥有物品
        3. PARTICIPATED_IN: 人物参与事件
        4. HAPPENED_AT: 事件发生在地点
        5. KNOWS: 人物之间的认识关系
        6. KILLED: 杀害关系
        7. SUSPECTS: 怀疑关系

        请以JSON格式返回关系列表，例如：
        [
            {{
                "source": "张三",
                "target": "客厅",
                "relation": "LOCATED_IN",
                "confidence": 0.9
            }}
        ]
        """

        request = LLMRequest(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.3
        )

        try:
            response = await self.generate_text(request)

            import json
            try:
                relations = json.loads(response.content)
                return relations if isinstance(relations, list) else []
            except json.JSONDecodeError:
                logger.warning("关系提取返回的不是有效JSON")
                return []

        except Exception as e:
            logger.error(f"关系提取失败: {str(e)}")
            raise LLMServiceError(f"关系提取失败: {str(e)}")

    async def text_to_ngql(self, question: str, schema_info: str = "") -> str:
        """
        将自然语言问题转换为nGQL查询语句

        Args:
            question: 自然语言问题
            schema_info: 图谱结构信息

        Returns:
            str: nGQL查询语句
        """
        prompt = f"""
        请将以下自然语言问题转换为NebulaGraph的nGQL查询语句：

        问题：{question}

        图谱结构信息：
        {schema_info}

        可用的标签类型：
        - Person(name string, role string)
        - Location(name string)
        - Item(name string, type string)
        - Event(name string, description string)
        - Timeline(name string, start string, end string)

        可用的边类型：
        - LOCATED_IN, HAS_ITEM, PARTICIPATED_IN, HAPPENED_AT, HAPPENED_ON

        请只返回nGQL查询语句，不要包含解释。
        例如：GO FROM "person1" OVER LOCATED_IN YIELD dst AS location
        """

        request = LLMRequest(
            prompt=prompt,
            max_tokens=500,
            temperature=0.1  # 很低的温度以确保准确性
        )

        try:
            response = await self.generate_text(request)
            # 清理响应，提取实际的nGQL语句
            ngql = response.content.strip()

            # 移除可能的markdown代码块标记
            if ngql.startswith("```"):
                ngql = ngql.split("\n", 1)[1]
            if ngql.endswith("```"):
                ngql = ngql.rsplit("\n", 1)[0]

            return ngql.strip()

        except Exception as e:
            logger.error(f"文本转nGQL失败: {str(e)}")
            raise LLMServiceError(f"文本转nGQL失败: {str(e)}")

    async def compress_text(self, text: str, target_ratio: float, preserve_elements: List[str] = None) -> str:
        """
        压缩文本

        Args:
            text: 原始文本
            target_ratio: 目标压缩比例
            preserve_elements: 必须保留的元素

        Returns:
            str: 压缩后的文本
        """
        prompt = f"""
        请将以下文本压缩到原来的{target_ratio:.0%}长度，同时保持核心内容和逻辑：

        原始文本：
        {text}

        必须保留的元素：
        {preserve_elements or '无特定要求'}

        压缩要求：
        1. 保持主要情节和人物关系
        2. 维持逻辑一致性
        3. 保留关键线索和转折点
        4. 语言要简洁流畅

        请返回压缩后的文本：
        """

        request = LLMRequest(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.5
        )

        try:
            response = await self.generate_text(request)
            return response.content.strip()

        except Exception as e:
            logger.error(f"文本压缩失败: {str(e)}")
            raise LLMServiceError(f"文本压缩失败: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态
        """
        try:
            # 发送一个简单的测试请求
            test_request = LLMRequest(
                prompt="请回答：1+1等于几？",
                max_tokens=10,
                temperature=0.1
            )

            response = await self.generate_text(test_request)

            return {
                "status": "healthy",
                "model": self.model,
                "response_time": "success",
                "test_response": response.content[:50] + "..." if len(response.content) > 50 else response.content
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model
            }

    def _parse_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """从文本中解析实体的备用方法"""
        # 简单的文本解析逻辑
        entities = {
            "persons": [],
            "locations": [],
            "items": [],
            "events": [],
            "times": []
        }

        # 这里可以实现更复杂的文本解析逻辑
        # 目前返回空字典，实际使用时需要完善
        return entities

    async def batch_extract_entities(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """
        批量提取实体

        Args:
            texts: 文本列表

        Returns:
            List[Dict[str, List[str]]]: 实体列表
        """
        tasks = [self.extract_entities(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量实体提取中遇到错误: {str(result)}")
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)

        return processed_results