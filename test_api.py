#!/usr/bin/env python3
"""
測試腳本：驗證劇本殺 Agent API 功能
"""

import requests
import json
import time


API_BASE = "http://localhost:9000"


def test_health():
    """测试健康检查"""
    print("=== 测试健康检查 ===")
    response = requests.get(f"{API_BASE}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()


def test_nebula_ping():
    """测试 NebulaGraph 连接"""
    print("=== 测试 NebulaGraph 连接 ===")
    response = requests.get(f"{API_BASE}/nebula/ping")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()


def test_nebula_query():
    """测试 NebulaGraph 查询"""
    print("=== 测试 NebulaGraph 查询 ===")
    query = {
        "nql": "SHOW SPACES;"
    }
    response = requests.post(f"{API_BASE}/nebula/query", json=query)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()


def test_text_to_ngql():
    """测试 Text-to-nGQL"""
    print("=== 测试 Text-to-nGQL ===")
    query = {
        "question": "王建国在哪里被发现死亡？"
    }
    response = requests.post(f"{API_BASE}/text-to-ngql", json=query)
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()


def test_extract_entities():
    """测试实体抽取"""
    print("=== 测试实体抽取 ===")
    
    # 读取样例剧本
    with open("sample_script.txt", "r", encoding="utf-8") as f:
        script_text = f.read()
    
    # 只取前500字符进行测试
    test_text = script_text[:500]
    
    query = {
        "text": test_text
    }
    response = requests.post(f"{API_BASE}/extract", json=query)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    print()


def test_rag_query():
    """测试 RAG 查询"""
    print("=== 测试 RAG 查询 ===")
    query = {
        "question": "王建国是怎么死的？"
    }
    response = requests.post(f"{API_BASE}/rag/query", json=query)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    print()


def test_vector_info():
    """测试向量库信息"""
    print("=== 测试向量库信息 ===")
    response = requests.get(f"{API_BASE}/vector/info")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()


def main():
    """主测试函数"""
    print("开始测试剧本杀 Agent API...")
    print("=" * 50)
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(10)
    
    try:
        # 基础功能测试
        test_health()
        test_nebula_ping()
        test_nebula_query()
        
        # 高级功能测试
        test_text_to_ngql()
        test_extract_entities()
        test_rag_query()
        test_vector_info()
        
        print("所有测试完成！")
        
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到 API 服务")
        print("请确保服务已启动：docker compose up -d")
    except Exception as e:
        print(f"测试过程中发生错误：{e}")


if __name__ == "__main__":
    main()
