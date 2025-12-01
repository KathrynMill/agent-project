"""应用配置管理模块"""

import os
from functools import lru_cache
from typing import Optional
from pydantic import BaseSettings, Field


class DatabaseSettings(BaseSettings):
    """数据库配置"""

    # NebulaGraph 配置
    nebula_host: str = Field(default="localhost", env="NEBULA_HOST")
    nebula_port: int = Field(default=9669, env="NEBULA_PORT")
    nebula_username: str = Field(default="root", env="NEBULA_USERNAME")
    nebula_password: str = Field(default="", env="NEBULA_PASSWORD")
    nebula_space_name: str = Field(default="scripts", env="NEBULA_SPACE_NAME")

    # Qdrant 配置
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field(default="script_embeddings", env="QDRANT_COLLECTION_NAME")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class LLMSettings(BaseSettings):
    """大语言模型配置"""

    # Google Gemini 配置
    gemini_api_key: str = Field(env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, env="GEMINI_TEMPERATURE")

    # vLLM 配置（本地部署）
    vllm_host: str = Field(default="localhost", env="VLLM_HOST")
    vllm_port: int = Field(default=8000, env="VLLM_PORT")
    vllm_model: str = Field(default="Qwen/Qwen2.5-14B-Instruct", env="VLLM_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class EmbeddingSettings(BaseSettings):
    """嵌入服务配置"""

    # BGE-M3 嵌入服务
    embedding_host: str = Field(default="localhost", env="EMBEDDING_HOST")
    embedding_port: int = Field(default=8080, env="EMBEDDING_PORT")
    embedding_model: str = Field(default="BAAI/bge-m3", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1024, env="EMBEDDING_DIMENSION")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ApplicationSettings(BaseSettings):
    """应用基础配置"""

    # 应用设置
    app_name: str = Field(default="剧本杀智能压缩系统", env="APP_NAME")
    app_version: str = Field(default="2.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")

    # API 配置
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=9000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")

    # 安全配置
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")

    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    # 性能配置
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class CompressionSettings(BaseSettings):
    """压缩系统配置"""

    # 压缩参数
    default_target_hours: int = Field(default=3, env="DEFAULT_TARGET_HOURS")
    max_iterations: int = Field(default=5, env="MAX_ITERATIONS")
    compression_ratio_min: float = Field(default=0.3, env="COMPRESSION_RATIO_MIN")
    compression_ratio_max: float = Field(default=0.8, env="COMPRESSION_RATIO_MAX")

    # 智能体配置
    agent_timeout: int = Field(default=120, env="AGENT_TIMEOUT")
    parallel_agents: bool = Field(default=True, env="PARALLEL_AGENTS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Settings:
    """全局配置管理器"""

    def __init__(self):
        self.database = DatabaseSettings()
        self.llm = LLMSettings()
        self.embedding = EmbeddingSettings()
        self.application = ApplicationSettings()
        self.compression = CompressionSettings()

    def validate_environment(self) -> bool:
        """验证环境配置"""
        try:
            # 检查必需的环境变量
            if not self.llm.gemini_api_key:
                raise ValueError("GEMINI_API_KEY 环境变量未设置")
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        return f"bolt://{self.database.nebula_host}:{self.database.nebula_port}"

    def get_llm_base_url(self) -> str:
        """获取LLM服务URL"""
        return f"http://{self.llm.vllm_host}:{self.llm.vllm_port}"

    def get_embedding_url(self) -> str:
        """获取嵌入服务URL"""
        return f"http://{self.embedding.embedding_host}:{self.embedding.embedding_port}"


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置实例（单例模式）"""
    return Settings()