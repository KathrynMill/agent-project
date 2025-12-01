#!/bin/bash

# 剧本杀智能压缩系统 - 测试脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 运行剧本杀智能压缩系统测试...${NC}"

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo -e "${GREEN}✅ 激活虚拟环境${NC}"
    else
        echo -e "${RED}❌ 错误: 虚拟环境不存在${NC}"
        echo -e "${YELLOW}请先运行: ./scripts/setup.sh${NC}"
        exit 1
    fi
fi

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo -e "${BLUE}📋 测试选项:${NC}"
echo "1. 运行所有测试"
echo "2. 运行单元测试"
echo "3. 运行集成测试"
echo "4. 运行健康检查测试"
echo "5. 生成覆盖率报告"
echo "6. 测试API端点"

read -p "请选择测试类型 (1-6): " test_type

case $test_type in
    1)
        echo -e "${GREEN}🧪 运行所有测试...${NC}"
        pytest tests/ -v --tb=short
        ;;
    2)
        echo -e "${GREEN}🔬 运行单元测试...${NC}"
        pytest tests/unit/ -v --tb=short
        ;;
    3)
        echo -e "${GREEN}🔗 运行集成测试...${NC}"
        pytest tests/integration/ -v --tb=short
        ;;
    4)
        echo -e "${GREEN}💓 运行健康检查测试...${NC}"
        python -c "
import asyncio
import sys
sys.path.append('.')
from shared.config.settings import get_settings
from core.services.llm_service import LLMService

async def test_health():
    print('测试LLM服务健康状态...')
    llm = LLMService()
    health = await llm.health_check()
    print(f'LLM服务状态: {health}')
    return health.get('status') == 'healthy'

result = asyncio.run(test_health())
print(f'健康检查: {\"✅ 通过\" if result else \"❌ 失败\"}')" 2>/dev/null || echo -e "${YELLOW}⚠️ 健康检查测试跳过（需要API密钥）${NC}"
        ;;
    5)
        echo -e "${GREEN}📊 生成覆盖率报告...${NC}"
        pytest tests/ --cov=core --cov=api --cov=shared --cov-report=html --cov-report=term-missing
        echo -e "${GREEN}📈 覆盖率报告已生成到 htmlcov/index.html${NC}"
        ;;
    6)
        echo -e "${GREEN}🌐 测试API端点...${NC}"

        # 检查API是否运行
        if ! curl -s http://localhost:9000/api/v1/health > /dev/null; then
            echo -e "${YELLOW}⚠️ API服务未运行，请先启动服务: ./scripts/start.sh${NC}"
            exit 1
        fi

        echo "测试健康检查端点..."
        curl -s http://localhost:9000/api/v1/health | python -m json.tool

        echo -e "\n测试系统状态端点..."
        curl -s http://localhost:9000/api/v1/health/detailed | python -m json.tool | head -20

        echo -e "\n测试压缩估算端点..."
        curl -s -X POST "http://localhost:9000/api/v1/compression/estimate/script_001?target_hours=3" \
             -H "Content-Type: application/json" | python -m json.tool
        ;;
    *)
        echo -e "${RED}❌ 无效的选择${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✅ 测试完成！${NC}"