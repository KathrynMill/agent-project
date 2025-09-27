#!/bin/bash

echo "🚀 启动剧本杀 Agent 系统..."

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    echo "❌ Docker 未运行，请先安装 Docker"
    echo "运行: ./install_docker.sh"
    exit 1
fi

# 检查 docker-compose 文件
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ 找不到 docker-compose.yml 文件"
    exit 1
fi

echo "📦 构建并启动所有服务..."
docker compose up -d --build

echo "⏳ 等待服务启动（约 2-3 分钟）..."
sleep 30

echo "🔍 检查服务状态..."
docker compose ps

echo "📊 检查服务健康状态..."
echo "等待 API 服务启动..."

# 等待 API 服务启动
for i in {1..30}; do
    if curl -s http://localhost:9000/health &> /dev/null; then
        echo "✅ API 服务已启动"
        break
    fi
    echo "等待中... ($i/30)"
    sleep 10
done

echo "🧪 运行测试..."
python3 test_api.py

echo "🎉 系统启动完成！"
echo ""
echo "📋 服务端点："
echo "- API 服务: http://localhost:9000"
echo "- vLLM 服务: http://localhost:8000"
echo "- 嵌入服务: http://localhost:8080"
echo "- Qdrant: http://localhost:6333"
echo "- NebulaGraph: localhost:9669"
echo ""
echo "🔧 管理命令："
echo "- 查看日志: docker compose logs -f"
echo "- 停止服务: docker compose down"
echo "- 重启服务: docker compose restart"
echo "- 查看状态: docker compose ps"






