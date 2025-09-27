#!/usr/bin/env python3
"""
多智能體系統測試腳本
測試劇本壓縮的多智能體工作流程
"""

import requests
import json
import time

# API 基礎 URL
BASE_URL = "http://localhost:9000"

def test_health():
    """測試健康檢查"""
    print("🔍 測試健康檢查...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ 健康檢查通過")
            return True
        else:
            print(f"❌ 健康檢查失敗: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康檢查異常: {str(e)}")
        return False

def test_compress_script():
    """測試劇本壓縮功能"""
    print("\n🎭 測試劇本壓縮功能...")
    
    # 準備測試數據
    test_data = {
        "player_scripts": {
            "王建國": """
            我是王建國，房地產大亨。今晚是我的生日晚宴，邀請了幾個朋友。
            我和妻子李美麗最近關係不太好，她總是懷疑我在外面有女人。
            其實我確實和劉秘書有關係，而且她懷了我的孩子。
            我準備修改遺囑，把大部分財產留給劉秘書。
            """,
            "李美麗": """
            我是李美麗，王建國的妻子。我知道他在外面有女人，但我選擇隱忍。
            今晚的晚宴上，我和他因為財產分配問題發生了激烈爭吵。
            我發現他準備把財產留給別的女人，這讓我非常憤怒。
            我利用他的心臟病，在爭吵中故意刺激他。
            """,
            "張律師": """
            我是張律師，王建國的法律顧問。我知道他準備修改遺囑。
            今晚的晚宴上，我試圖勸解王建國和李美麗的爭吵。
            我看到了王建國的遺囑草稿，他準備把大部分財產留給劉秘書。
            這讓我感到震驚，因為這會影響到李美麗的利益。
            """,
            "陳醫生": """
            我是陳醫生，王建國的私人醫生。我知道他有心臟病，但沒有告訴其他人。
            今晚的晚宴上，我去洗手間時發現王建國倒在書房的地板上。
            我立即檢查，發現他已經死亡，胸口插著一把水果刀。
            我注意到他手中握著一張寫著"背叛者"的紙條。
            """,
            "劉秘書": """
            我是劉秘書，王建國的秘書。我們有秘密關係，我懷了他的孩子。
            今晚的晚宴上，我的手機在書房內，最後通話是晚上8點45分。
            我知道王建國準備修改遺囑，把財產留給我。
            這讓我既興奮又擔心，因為這會引起很多人的不滿。
            """
        },
        "master_guide": """
        這是一個密室殺人案。王建國在書房被發現死亡，書房門從內部鎖住。
        關鍵線索：
        1. 王建國手中握著"背叛者"紙條
        2. 書房內發現李美麗的發夾
        3. 張律師的公文包在書房門口
        4. 陳醫生的聽診器在書房內
        5. 劉秘書的手機在書房內
        
        真相：李美麗是凶手。她利用王建國的心臟病，在爭吵中故意刺激他，
        導致他心臟病發作死亡。然後她進入書房，用水果刀刺向已經死亡的王建國，
        製造他殺的假象，並留下"背叛者"紙條暗示是劉秘書所為。
        """,
        "target_hours": 2
    }
    
    try:
        print("📤 發送壓縮請求...")
        response = requests.post(
            f"{BASE_URL}/compress_script",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 劇本壓縮成功")
            
            # 顯示結果摘要
            print("\n📊 壓縮結果摘要:")
            if "compression_result" in result:
                compression_result = result["compression_result"]
                
                # 顯示壓縮後的劇本
                if "compressed_script" in compression_result:
                    print(f"\n📝 壓縮後的劇本:")
                    print("-" * 50)
                    print(compression_result["compressed_script"])
                    print("-" * 50)
                
                # 顯示壓縮比例
                if "compression_ratio" in compression_result:
                    print(f"\n📈 壓縮比例: {compression_result['compression_ratio']:.1%}")
                
                # 顯示驗證結果
                if "validation_summary" in compression_result:
                    validation = compression_result["validation_summary"]
                    print(f"\n✅ 驗證結果:")
                    print(f"   總體分數: {validation.get('overall_score', 'N/A')}")
                    print(f"   驗證狀態: {validation.get('validation_status', 'N/A')}")
                    if validation.get('critical_issues'):
                        print(f"   關鍵問題: {', '.join(validation['critical_issues'])}")
                
                # 顯示智能體工作流程
                if "agent_workflow" in compression_result:
                    workflow = compression_result["agent_workflow"]
                    print(f"\n🤖 智能體工作流程:")
                    if "analysis_result" in workflow:
                        print("   ✅ 分析智能體完成")
                    if "compression_strategy" in workflow:
                        print("   ✅ 策略制定完成")
                    if "debate_result" in workflow:
                        print("   ✅ 智能體辯論完成")
                    if "validation_result" in workflow:
                        print("   ✅ 校驗智能體完成")
            
            return True
        else:
            print(f"❌ 劇本壓縮失敗: {response.status_code}")
            print(f"錯誤詳情: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 劇本壓縮異常: {str(e)}")
        return False

def test_individual_agents():
    """測試個別智能體功能"""
    print("\n🤖 測試個別智能體功能...")
    
    # 這裡可以添加對個別智能體的測試
    # 例如直接調用 AnalysisAgent、LogicAgent 等
    print("ℹ️  個別智能體測試需要直接調用，暫時跳過")

def main():
    """主測試函數"""
    print("🚀 開始多智能體系統測試")
    print("=" * 60)
    
    # 測試健康檢查
    if not test_health():
        print("❌ 健康檢查失敗，停止測試")
        return
    
    # 等待服務完全啟動
    print("\n⏳ 等待服務完全啟動...")
    time.sleep(5)
    
    # 測試劇本壓縮
    if test_compress_script():
        print("\n🎉 多智能體系統測試完成！")
    else:
        print("\n❌ 多智能體系統測試失敗")
    
    print("\n" + "=" * 60)
    print("測試完成")

if __name__ == "__main__":
    main()
