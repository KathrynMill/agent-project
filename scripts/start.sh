#!/bin/bash

# 剧本杀智能压缩系统 - 启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 启动剧本杀智能压缩系统...${NC}"

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  未检测到虚拟环境，正在创建...${NC}"

    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    source venv/bin/activate
    echo -e "${GREEN}✅ 虚拟环境已激活${NC}"
else
    echo -e "${GREEN}✅ 已在虚拟环境中: $VIRTUAL_ENV${NC}"
fi

# 检查.env文件
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ 错误: .env 文件不存在${NC}"
    echo -e "${YELLOW}请先运行: ./scripts/setup.sh${NC}"
    exit 1
fi

# 加载环境变量
export $(grep -v '^#' .env | xargs)

# 检查必要的环境变量
required_vars=("GEMINI_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo -e "${RED}❌ 错误: 以下环境变量未设置: ${missing_vars[*]}${NC}"
    echo -e "${YELLOW}请编辑 .env 文件并设置这些变量${NC}"
    exit 1
fi

# 创建日志目录
mkdir -p logs

echo -e "${BLUE}📋 启动选项:${NC}"
echo "1. 开发模式 (热重载)"
echo "2. 生产模式"
echo "3. 仅启动API服务"
echo "4. 启动外部服务 + API"

read -p "请选择启动模式 (1-4): " mode

case $mode in
    1)
        echo -e "${GREEN}🔧 启动开发模式...${NC}"
        export PYTHONPATH="${PYTHONPATH}:$(pwd)"
        uvicorn api.app:app \
            --host ${API_HOST:-0.0.0.0} \
            --port ${API_PORT:-9000} \
            --reload \
            --log-level ${LOG_LEVEL:-info} \
            --reload-dir core \
            --reload-dir api \
            --reload-dir shared
        ;;
    2)
        echo -e "${GREEN}🚀 启动生产模式...${NC}"
        export PYTHONPATH="${PYTHONPATH}:$(pwd)"
        gunicorn -w ${WORKERS:-4} \
            -k uvicorn.workers.UvicornWorker \
            --bind ${API_HOST:-0.0.0.0}:${API_PORT:-9000} \
            --log-level ${LOG_LEVEL:-info} \
            --access-logfile logs/access.log \
            --error-logfile logs/error.log \
            --timeout ${REQUEST_TIMEOUT:-300} \
            api.app:app
        ;;
    3)
        echo -e "${GREEN}🌐 仅启动API服务...${NC}"
        export PYTHONPATH="${PYTHONPATH}:$(pwd)"
        uvicorn api.app:app \
            --host ${API_HOST:-0.0.0.0} \
            --port ${API_PORT:-9000} \
            --log-level ${LOG_LEVEL:-info}
        ;;
    4)
        echo -e "${GREEN}🐳 启动外部服务 + API...${NC}"

        # 启动NebulaGraph
        if ! docker ps | grep -q nebula; then
            echo "启动 NebulaGraph..."
            docker run -d --name nebula \
                -p 9669:9669 \
                -p 19669:19669 \
                -p 19670:19670 \
                vesoft/nebula-graph:v3.5.0
        else
            echo "✅ NebulaGraph 已在运行"
        fi

        # 启动Qdrant
        if ! docker ps | grep -q qdrant; then
            echo "启动 Qdrant..."
            docker run -d --name qdrant \
                -p 6333:6333 \
                -p 6334:6334 \
                qdrant/qdrant:latest
        else
            echo "✅ Qdrant 已在运行"
        fi

        # 等待服务启动
        echo "等待外部服务启动..."
        sleep 10

        # 启动API
        export PYTHONPATH="${PYTHONPATH}:$(pwd)"
        uvicorn api.app:app \
            --host ${API_HOST:-0.0.0.0} \
            --port ${API_PORT:-9000} \
            --reload \
            --log-level ${LOG_LEVEL:-info}
        ;;
    *)
        echo -e "${RED}❌ 无效的选择${NC}"
        exit 1
        ;;
esac