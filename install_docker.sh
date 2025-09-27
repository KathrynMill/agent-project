#!/bin/bash

echo "🚀 开始安装 Docker 和运行剧本杀 Agent 系统..."

# 检查是否为 root 用户
if [ "$EUID" -eq 0 ]; then
    echo "❌ 请不要使用 root 用户运行此脚本"
    exit 1
fi

# 检查系统
echo "📋 检查系统信息..."
lsb_release -a

# 更新包列表
echo "🔄 更新包列表..."
sudo apt update

# 安装必要的依赖
echo "📦 安装必要依赖..."
sudo apt install -y curl wget git

# 检查是否已安装 Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker 已安装"
else
    echo "📦 安装 Docker..."
    sudo apt install -y docker.io docker-compose
fi

# 启动 Docker 服务
echo "🔧 启动 Docker 服务..."
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到 docker 组
echo "👤 添加用户到 docker 组..."
sudo usermod -aG docker $USER

# 检查 Docker 安装
echo "✅ 验证 Docker 安装..."
docker --version
docker-compose --version

echo "🎉 Docker 安装完成！"
echo "⚠️  请重新登录或重启终端，然后运行: ./run_system.sh"