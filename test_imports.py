#!/usr/bin/env python3
"""
测试模块导入的简单脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试关键模块导入"""
    print("🧪 测试模块导入...")

    try:
        # 测试基础模块
        print("1. 测试配置模块...")
        from shared.config.settings import get_settings
        print("   ✅ 配置模块导入成功")

        # 测试数据模型
        print("2. 测试数据模型...")
        from core.models.script_models import Script, ScriptMetadata
        from core.models.compression_models import CompressionRequest
        print("   ✅ 数据模型导入成功")

        # 测试智能体基类
        print("3. 测试智能体模块...")
        from core.agents.base_agent import BaseAgent, AgentResponse
        print("   ✅ 智能体模块导入成功")

        # 测试服务模块
        print("4. 测试服务模块...")
        from core.services.llm_service import LLMService
        print("   ✅ LLM服务模块导入成功")

        print("\n🎉 所有模块导入测试通过！")
        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_basic_functionality():
    """测试基础功能"""
    print("\n🔧 测试基础功能...")

    try:
        # 测试配置
        from shared.config.settings import get_settings
        settings = get_settings()
        print(f"   ✅ 配置加载成功 - 应用: {settings.application.app_name}")

        # 测试数据模型创建
        from core.models.script_models import ScriptMetadata, Script
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

        # 测试智能体创建
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
        print(f"   ✅ 智能体创建成功 - 名称: {agent.name}")

        print("\n✅ 所有基础功能测试通过！")
        return True

    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 剧本杀智能压缩系统 - 导入测试")
    print("=" * 50)

    # 测试导入
    import_ok = test_imports()

    if import_ok:
        # 测试基础功能
        functionality_ok = test_basic_functionality()

        if functionality_ok:
            print("\n🎉 系统基础功能正常！")
            print("\n📋 下一步:")
            print("1. 配置 .env 文件中的 GEMINI_API_KEY")
            print("2. 启动开发服务器: python -m api.app")
            print("3. 访问 API 文档: http://localhost:9000/docs")
        else:
            print("\n❌ 基础功能测试失败")
    else:
        print("\n❌ 模块导入测试失败")
        print("💡 请确保依赖已正确安装:")
        print("   source venv/bin/activate")
        print("   pip install -r requirements/base.txt")

if __name__ == "__main__":
    main()