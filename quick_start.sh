#!/bin/bash

echo "🚀 剧本杀 Agent 快速启动脚本"
echo "================================"

# 检查 Docker 是否可用
if command -v docker &> /dev/null; then
    echo "✅ Docker 已安装"
    docker --version
else
    echo "❌ Docker 未安装"
    echo "请先运行: ./install_docker.sh"
    echo "或者手动安装 Docker:"
    echo "sudo apt update && sudo apt install docker.io docker-compose"
    exit 1
fi

# 检查 docker-compose 是否可用
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose 已安装"
    docker-compose --version
else
    echo "❌ Docker Compose 未安装"
    echo "请安装: sudo apt install docker-compose"
    exit 1
fi

# 检查当前用户是否在 docker 组中
if groups $USER | grep -q docker; then
    echo "✅ 用户已在 docker 组中"
else
    echo "⚠️  用户不在 docker 组中，可能需要使用 sudo"
    echo "建议运行: sudo usermod -aG docker $USER"
    echo "然后重新登录"
fi

echo ""
echo "🔧 启动服务..."

# 尝试启动服务
if docker-compose up -d --build; then
    echo "✅ 服务启动成功"
else
    echo "❌ 服务启动失败，尝试使用 sudo..."
    if sudo docker-compose up -d --build; then
        echo "✅ 使用 sudo 启动成功"
    else
        echo "❌ 启动失败，请检查错误信息"
        exit 1
    fi
fi

echo ""
echo "⏳ 等待服务启动..."
sleep 30

echo "🔍 检查服务状态..."
docker-compose ps

echo ""
echo "🧪 测试 API 服务..."
if curl -s http://localhost:9000/health; then
    echo "✅ API 服务正常"
else
    echo "⚠️  API 服务可能还在启动中，请稍等..."
fi

echo ""
echo "🎉 启动完成！"
echo "访问 http://localhost:9000 查看 API 文档"
echo "运行 python3 test_api.py 进行完整测试"






