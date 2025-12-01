#!/bin/bash

# 剧本杀智能压缩系统 - 环境设置脚本

set -e

echo "🚀 开始设置剧本杀智能压缩系统..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要Python 3.8或更高版本，当前版本: $python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 检查并创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建Python虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 安装项目依赖..."
pip install -r requirements/dev.txt

# 检查并创建.env文件
if [ ! -f ".env" ]; then
    echo "📝 创建环境配置文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件，填入必要的配置（特别是API密钥）"
else
    echo "✅ 环境配置文件已存在"
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p logs
mkdir -p data
mkdir -p temp
mkdir -p tests/fixtures/sample_scripts

# 检查Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker 已安装"
    if command -v docker-compose &> /dev/null; then
        echo "✅ Docker Compose 已安装"
    else
        echo "⚠️  Docker Compose 未安装，建议安装以便运行外部服务"
    fi
else
    echo "⚠️  Docker 未安装，某些功能可能无法使用"
fi

# 检查基础依赖
echo "🔍 检查Python依赖..."
python -c "import fastapi, uvicorn, pydantic" && echo "✅ 基础Web框架依赖正常"
python -c "import nebula3, qdrant_client" && echo "✅ 数据库客户端依赖正常" || echo "⚠️  数据库客户端可能需要额外配置"

# 设置权限
chmod +x scripts/*.sh

echo ""
echo "🎉 环境设置完成！"
echo ""
echo "📋 下一步操作："
echo "1. 编辑 .env 文件，配置必要的环境变量"
echo "2. 启动外部服务（可选）："
echo "   - NebulaGraph: docker run -d --name nebula -p 9669:9669 vesoft/nebula-graph:v3.5.0"
echo "   - Qdrant: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest"
echo "3. 运行系统: ./scripts/start.sh"
echo ""
echo "📖 更多信息请查看 README.md"