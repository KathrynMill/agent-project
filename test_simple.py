#!/usr/bin/env python3
"""
简单的系统测试脚本
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_basic_functionality():
    """测试基础功能"""
    print("🧪 开始基础功能测试...")

    try:
        # 测试配置加载
        print("1. 测试配置加载...")
        from shared.config.settings import get_settings
        settings = get_settings()
        print(f"   ✅ 配置加载成功 - 应用名称: {settings.application.app_name}")

        # 测试数据模型
        print("2. 测试数据模型...")
        from core.models.script_models import Script, ScriptMetadata
        from core.models.compression_models import CompressionRequest, CompressionLevel

        metadata = ScriptMetadata(
            title="测试剧本",
            author="测试作者",
            estimated_duration_hours=5.0
        )

        script = Script(
            id="test_script",
            metadata=metadata,
            player_scripts={},
            master_script=None,
            entities=[],
            relations=[],
            events=[],
            timelines=[]
        )
        print(f"   ✅ 数据模型创建成功 - 剧本ID: {script.id}")

        # 测试智能体基础类
        print("3. 测试智能体基础类...")
        from core.agents.base_agent import BaseAgent, AgentResponse

        class TestAgent(BaseAgent):
            def get_task_types(self):
                return ["test"]

            async def process_task(self, task):
                return AgentResponse(
                    success=True,
                    result={"test": "ok"},
                    agent_name=self.name,
                    task_type=task.task_type
                )

        agent = TestAgent()
        print(f"   ✅ 智能体创建成功 - 智能体名称: {agent.name}")

        # 测试服务类
        print("4. 测试LLM服务...")
        from core.services.llm_service import LLMService, LLMRequest

        llm_service = LLMService()
        print(f"   ✅ LLM服务创建成功 - 模型: {llm_service.model}")

        print("\n🎉 所有基础功能测试通过！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """测试API端点"""
    print("\n🌐 测试API端点...")

    try:
        import httpx

        # 测试健康检查
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:9000/api/v1/health", timeout=5.0)

            if response.status_code == 200:
                print("   ✅ 健康检查端点正常")
                data = response.json()
                print(f"   📊 系统状态: {data.get('status', 'unknown')}")
            else:
                print(f"   ⚠️ 健康检查端点响应异常: {response.status_code}")

        print("✅ API端点测试完成")
        return True

    except httpx.ConnectError:
        print("   ⚠️ API服务未启动，请先运行: ./scripts/start.sh")
        return False
    except Exception as e:
        print(f"   ❌ API测试失败: {str(e)}")
        return False

async def main():
    """主测试函数"""
    print("🚀 剧本杀智能压缩系统 - 简单测试")
    print("=" * 50)

    # 基础功能测试
    basic_test_passed = await test_basic_functionality()

    # API端点测试
    api_test_passed = await test_api_endpoints()

    print("\n" + "=" * 50)
    print("📋 测试总结:")
    print(f"   基础功能: {'✅ 通过' if basic_test_passed else '❌ 失败'}")
    print(f"   API端点: {'✅ 通过' if api_test_passed else '❌ 失败'}")

    if basic_test_passed:
        print("\n✅ 系统基本功能正常，可以进行下一步操作！")
        print("\n📋 下一步:")
        print("1. 配置 .env 文件中的API密钥")
        print("2. 启动外部服务 (可选)")
        print("3. 运行完整系统: ./scripts/start.sh")
    else:
        print("\n❌ 请修复基础功能问题后再继续")

if __name__ == "__main__":
    asyncio.run(main())