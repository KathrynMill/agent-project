#!/usr/bin/env python3
"""
快速启动脚本 - 剧本杀智能压缩系统
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def print_banner():
    """打印启动横幅"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🎭 剧本杀智能压缩系统 V2.1 - 快速启动                ║
║                                                              ║
║     基于多智能体的剧本杀剧本智能压缩系统                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    print(f"✅ Python版本检查通过: {sys.version}")
    return True

def check_environment():
    """检查环境"""
    # 检查虚拟环境
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    if not in_venv:
        print("⚠️  建议在虚拟环境中运行")

        # 创建虚拟环境
        venv_path = Path("venv")
        if not venv_path.exists():
            print("📦 创建虚拟环境...")
            result = subprocess.run([sys.executable, "-m", "venv", "venv"], capture_output=True)
            if result.returncode != 0:
                print("❌ 创建虚拟环境失败")
                return False
            print("✅ 虚拟环境创建成功")

        print("💡 激活虚拟环境:")
        print("   Linux/Mac: source venv/bin/activate")
        print("   Windows: venv\\Scripts\\activate")
        print("   然后重新运行此脚本")
        return False

    print("✅ 虚拟环境检查通过")
    return True

def check_dependencies():
    """检查依赖"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ 核心依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("💡 运行以下命令安装依赖:")
        print("   pip install -r requirements/dev.txt")
        return False

def check_config():
    """检查配置"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env 文件不存在")

        # 从模板创建
        template_file = Path(".env.example")
        if template_file.exists():
            print("📝 从模板创建 .env 文件...")
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ .env 文件创建成功")
            print("⚠️  请编辑 .env 文件，设置必要的环境变量（特别是 GEMINI_API_KEY）")
        else:
            print("❌ .env.example 文件不存在")
            return False

    print("✅ 配置文件检查通过")
    return True

async def run_basic_test():
    """运行基础测试"""
    print("\n🧪 运行基础功能测试...")

    try:
        # 测试导入
        print("   测试模块导入...")
        from shared.config.settings import get_settings
        from core.models.script_models import Script, ScriptMetadata
        from core.agents.base_agent import BaseAgent
        from core.services.llm_service import LLMService

        print("   ✅ 所有模块导入成功")

        # 测试配置
        settings = get_settings()
        print(f"   ✅ 配置加载成功 - {settings.application.app_name}")

        # 测试数据模型
        metadata = ScriptMetadata(
            title="测试剧本",
            author="测试作者",
            estimated_duration_hours=5.0
        )
        script = Script(
            id="test",
            metadata=metadata,
            player_scripts={},
            master_script=None,
            entities=[],
            relations=[],
            events=[],
            timelines=[]
        )
        print(f"   ✅ 数据模型创建成功 - 剧本: {script.title}")

        return True

    except Exception as e:
        print(f"   ❌ 基础测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_next_steps():
    """显示下一步操作"""
    print("\n" + "="*60)
    print("🎉 系统检查完成！")
    print("\n📋 下一步操作:")
    print("1. 📝 编辑 .env 文件，设置 GEMINI_API_KEY")
    print("2. 🚀 启动系统:")
    print("   python -m api.app")
    print("   或使用: ./scripts/start.sh")
    print("3. 🌐 访问API文档:")
    print("   http://localhost:9000/docs")
    print("4. 🧪 运行测试:")
    print("   python test_simple.py")
    print("   或使用: ./scripts/test.sh")
    print("\n📖 更多信息请查看 README.md")

def start_development_server():
    """启动开发服务器"""
    print("\n🚀 是否立即启动开发服务器? (y/n): ", end="")
    choice = input().strip().lower()

    if choice in ['y', 'yes', '是']:
        print("启动开发服务器...")
        try:
            # 设置PYTHONPATH
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}"

            # 启动uvicorn
            subprocess.run([
                sys.executable, "-m", "uvicorn",
                "api.app:app",
                "--host", "0.0.0.0",
                "--port", "9000",
                "--reload",
                "--log-level", "info"
            ], env=env)
        except KeyboardInterrupt:
            print("\n👋 服务已停止")
        except Exception as e:
            print(f"❌ 启动失败: {e}")

async def main():
    """主函数"""
    print_banner()

    print("🔍 系统环境检查...")
    print("-" * 40)

    # 环境检查
    checks = [
        ("Python版本", check_python_version),
        ("虚拟环境", check_environment),
        ("依赖包", check_dependencies),
        ("配置文件", check_config),
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"检查 {name}...")
        if not check_func():
            all_passed = False
            print(f"❌ {name} 检查失败")
            break

    if not all_passed:
        print("\n💡 请解决上述问题后重新运行")
        return

    # 运行基础测试
    if not await run_basic_test():
        print("\n💡 请修复基础功能问题")
        return

    show_next_steps()
    start_development_server()

if __name__ == "__main__":
    asyncio.run(main())