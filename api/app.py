"""剧本杀智能压缩系统 - FastAPI 主应用"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routers import compression, rag, health, admin
from ..shared.config.settings import get_settings
from ..shared.utils.exceptions import BaseSystemException

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 获取配置
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("剧本杀智能压缩系统启动中...")
    logger.info(f"环境: {'开发' if settings.application.debug else '生产'}")
    logger.info(f"API地址: http://{settings.application.api_host}:{settings.application.api_port}")

    # 这里可以添加初始化逻辑，比如数据库连接检查等
    try:
        # 初始化服务
        from ..core.services.compression_service import CompressionService
        app.state.compression_service = CompressionService()

        logger.info("所有服务初始化完成")
    except Exception as e:
        logger.error(f"服务初始化失败: {str(e)}")
        raise

    yield

    # 关闭时执行
    logger.info("剧本杀智能压缩系统关闭中...")
    # 这里可以添加清理逻辑


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title=settings.application.app_name,
        version=settings.application.app_version,
        description="基于专用AI模型的剧本杀剧本智能压缩系统",
        lifespan=lifespan,
        debug=settings.application.debug
    )

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.application.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 添加异常处理器
    app.add_exception_handler(BaseSystemException, system_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # 添加请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"{request.method} {request.url.path} - "
            f"状态码: {response.status_code} - "
            f"处理时间: {process_time:.3f}s"
        )
        return response

    # 注册路由
    app.include_router(
        compression.router,
        prefix=settings.application.api_prefix,
        tags=["压缩服务"]
    )
    app.include_router(
        rag.router,
        prefix=settings.application.api_prefix,
        tags=["RAG问答"]
    )
    app.include_router(
        health.router,
        prefix=settings.application.api_prefix,
        tags=["健康检查"]
    )
    app.include_router(
        admin.router,
        prefix=settings.application.api_prefix,
        tags=["系统管理"]
    )

    return app


# 创建应用实例
app = create_app()


# 异常处理器
async def system_exception_handler(request: Request, exc: BaseSystemException):
    """系统异常处理器"""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.to_dict(),
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "服务器内部错误",
                "details": str(exc) if settings.application.debug else None
            },
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )


# 根路径
@app.get("/")
async def root():
    """根路径信息"""
    return {
        "name": settings.application.app_name,
        "version": settings.application.app_version,
        "description": "基于专用AI模型的剧本杀剧本智能压缩系统",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }


if __name__ == "__main__":
    import time
    uvicorn.run(
        "api.app:app",
        host=settings.application.api_host,
        port=settings.application.api_port,
        reload=settings.application.debug,
        log_level=settings.application.log_level.lower()
    )