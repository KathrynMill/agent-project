#!/usr/bin/env python3
"""
系統測試腳本 - 驗證多智能體系統的基本功能
"""

import sys
import os
import asyncio
import json

# 添加項目路徑
sys.path.append('/home/aa/agent-project/api')

async def test_basic_functionality():
    """測試基本功能"""
    print("開始系統測試...")
    print("=" * 50)
    
    try:
        # 測試服務層導入和初始化
        print("1. 測試服務層...")
        from app.services.llm_service import LLMService
        from app.services.nebula_service import NebulaService
        from app.services.vector_service import VectorService
        
        llm_service = LLMService()
        nebula_service = NebulaService()
        vector_service = VectorService()
        print("✓ 服務層初始化成功")
        
        # 測試智能體導入和初始化
        print("\n2. 測試智能體層...")
        from app.agents.analysis_agent import AnalysisAgent
        from app.agents.logic_agent import LogicAgent
        from app.agents.story_agent import StoryAgent
        from app.agents.validation_agent import ValidationAgent
        from app.agents.chief_editor_agent import ChiefEditorAgent
        
        analysis_agent = AnalysisAgent(nebula_service, llm_service)
        logic_agent = LogicAgent(llm_service)
        story_agent = StoryAgent(llm_service)
        validation_agent = ValidationAgent(nebula_service, llm_service)
        chief_editor = ChiefEditorAgent(nebula_service, llm_service)
        print("✓ 智能體層初始化成功")
        
        # 測試主應用導入
        print("\n3. 測試主應用...")
        from app.main import app
        print("✓ 主應用導入成功")
        
        # 測試基本功能
        print("\n4. 測試基本功能...")
        
        # 測試 LLM 服務
        response = await llm_service.generate_text("測試提示")
        print(f"✓ LLM 服務響應: {response[:50]}...")
        
        # 測試 NebulaGraph 服務
        result = nebula_service.execute_query("SHOW SPACES")
        print(f"✓ NebulaGraph 服務響應: {result}")
        
        # 測試向量服務
        vector_result = await vector_service.search_similar("測試查詢", 5)
        print(f"✓ 向量服務響應: {vector_result}")
        
        # 測試工作流程創建
        print("\n5. 測試工作流程創建...")
        workflow = await chief_editor.create_compression_workflow()
        print("✓ 工作流程創建成功")
        
        # 測試壓縮流程
        print("\n6. 測試壓縮流程...")
        test_input = {
            "player_scripts": {
                "角色A": "我是角色A，我看到了重要線索",
                "角色B": "我是角色B，我有不在場證明"
            },
            "master_guide": "這是一個測試劇本",
            "target_hours": 2
        }
        
        result = await workflow.invoke(test_input)
        print(f"✓ 壓縮流程執行成功")
        print(f"結果類型: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """測試 API 端點"""
    print("\n7. 測試 API 端點...")
    
    try:
        from app.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # 測試健康檢查
        response = client.get("/health")
        print(f"✓ 健康檢查: {response.status_code}")
        
        # 測試壓縮端點
        test_data = {
            "player_scripts": {
                "角色A": "我是角色A的劇本",
                "角色B": "我是角色B的劇本"
            },
            "master_guide": "主持人手冊",
            "target_hours": 2
        }
        
        response = client.post("/compress_script", json=test_data)
        print(f"✓ 壓縮端點: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"響應內容: {json.dumps(result, ensure_ascii=False, indent=2)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ API 測試失敗: {str(e)}")
        return False

async def main():
    """主測試函數"""
    print("多智能體劇本壓縮系統測試")
    print("=" * 50)
    
    # 基本功能測試
    basic_success = await test_basic_functionality()
    
    # API 測試
    api_success = await test_api_endpoints()
    
    print("\n" + "=" * 50)
    if basic_success and api_success:
        print("✓ 所有測試通過！系統可以正常運行")
        return 0
    else:
        print("✗ 部分測試失敗，需要修復")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))


