"""自定义异常类模块"""


class BaseSystemException(Exception):
    """系统基础异常类"""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(BaseSystemException):
    """配置错误"""
    pass


class DatabaseError(BaseSystemException):
    """数据库操作错误"""
    pass


class LLMServiceError(BaseSystemException):
    """大语言模型服务错误"""
    pass


class EmbeddingServiceError(BaseSystemException):
    """嵌入服务错误"""
    pass


class ValidationError(BaseSystemException):
    """数据验证错误"""
    pass


class ProcessingError(BaseSystemException):
    """数据处理错误"""
    pass


class CompressionError(BaseSystemException):
    """剧本压缩错误"""
    pass


class AgentError(BaseSystemException):
    """智能体错误"""
    pass


class WorkflowError(BaseSystemException):
    """工作流错误"""
    pass


class TimeoutError(BaseSystemException):
    """超时错误"""
    pass


class ResourceExhaustedError(BaseSystemException):
    """资源耗尽错误"""
    pass


class AuthenticationError(BaseSystemException):
    """认证错误"""
    pass


class AuthorizationError(BaseSystemException):
    """授权错误"""
    pass


class RateLimitError(BaseSystemException):
    """频率限制错误"""
    pass


class ExternalServiceError(BaseSystemException):
    """外部服务错误"""
    pass


class ParsingError(BaseSystemException):
    """解析错误"""
    pass


class LogicError(BaseSystemException):
    """逻辑错误"""
    pass


# 具体的异常子类


class ScriptParsingError(ParsingError):
    """剧本解析错误"""
    pass


class EntityExtractionError(ProcessingError):
    """实体提取错误"""
    pass


class RelationExtractionError(ProcessingError):
    """关系提取错误"""
    pass


class NebulaGraphError(DatabaseError):
    """NebulaGraph 数据库错误"""
    pass


class QdrantError(DatabaseError):
    """Qdrant 向量数据库错误"""
    pass


class AgentTimeoutError(TimeoutError):
    """智能体超时错误"""
    pass


class AgentExecutionError(AgentError):
    """智能体执行错误"""
    pass


class CompressionValidationError(ValidationError):
    """压缩验证错误"""
    pass


class WorkflowStateError(WorkflowError):
    """工作流状态错误"""
    pass


class ModelInferenceError(LLMServiceError):
    """模型推理错误"""
    pass


class TextTooLongError(ValidationError):
    """文本过长错误"""
    pass


class InsufficientDataError(ProcessingError):
    """数据不足错误"""
    pass


class InconsistentDataError(LogicError):
    """数据不一致错误"""
    pass


class DependencyError(BaseSystemException):
    """依赖错误"""
    pass


class ServiceUnavailableError(ExternalServiceError):
    """服务不可用错误"""
    pass


# 异常工具函数


def handle_database_error(func):
    """数据库操作错误装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise DatabaseError(f"数据库操作失败: {str(e)}", "DB_OPERATION_FAILED")
    return wrapper


def handle_llm_error(func):
    """LLM服务错误装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise LLMServiceError(f"LLM服务调用失败: {str(e)}", "LLM_SERVICE_FAILED")
    return wrapper


def handle_agent_error(func):
    """智能体错误装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise AgentError(f"智能体执行失败: {str(e)}", "AGENT_EXECUTION_FAILED")
    return wrapper


def handle_validation_error(func):
    """验证错误装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            raise ValidationError(f"数据验证失败: {str(e)}", "VALIDATION_FAILED")
        except Exception as e:
            raise ValidationError(f"验证过程异常: {str(e)}", "VALIDATION_ERROR")
    return wrapper


# 错误代码常量
class ErrorCodes:
    """错误代码常量"""

    # 通用错误
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    BAD_REQUEST = "BAD_REQUEST"

    # 配置错误
    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"

    # 数据库错误
    DB_CONNECTION_FAILED = "DB_CONNECTION_FAILED"
    DB_QUERY_FAILED = "DB_QUERY_FAILED"
    DB_TRANSACTION_FAILED = "DB_TRANSACTION_FAILED"

    # LLM服务错误
    LLM_API_KEY_INVALID = "LLM_API_KEY_INVALID"
    LLM_QUOTA_EXCEEDED = "LLM_QUOTA_EXCEEDED"
    LLM_MODEL_NOT_FOUND = "LLM_MODEL_NOT_FOUND"
    LLM_INFERENCE_FAILED = "LLM_INFERENCE_FAILED"

    # 智能体错误
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"

    # 压缩错误
    COMPRESSION_FAILED = "COMPRESSION_FAILED"
    COMPRESSION_VALIDATION_FAILED = "COMPRESSION_VALIDATION_FAILED"
    COMPRESSION_RATIO_INVALID = "COMPRESSION_RATIO_INVALID"

    # 数据错误
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    DATA_INVALID = "DATA_INVALID"
    DATA_TOO_LARGE = "DATA_TOO_LARGE"
    DATA_CORRUPTED = "DATA_CORRUPTED"

    # 认证授权错误
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_INVALID = "TOKEN_INVALID"