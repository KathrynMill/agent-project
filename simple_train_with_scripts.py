#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版训练脚本 - 使用二分之一推理世界完整版2目录下的剧本文件
此脚本直接处理文本数据并提供基本分析，不依赖Docker服务
"""

import os
import sys
import re
from collections import Counter

# 配置路径
# SCRIPTS_DIR = r"c:\Users\11928\Desktop\linshi\1"
SCRIPTS_DIR = r"c:\Users\11928\Desktop\linshi\二分之一推理世界完整版2"
OUTPUT_DIR = r"c:\Users\11928\Desktop\linshi\output"
SCRIPTS_FILE = os.path.join(OUTPUT_DIR, "merged_scripts.txt")
MANUALS_FILE = os.path.join(OUTPUT_DIR, "merged_manuals.txt")


def check_and_process_docx_files():
    """检查并处理DOCX文件"""
    print(f"🔍 检查 {SCRIPTS_DIR} 目录中的DOCX文件...")
    
    # 检查源目录是否存在
    if not os.path.exists(SCRIPTS_DIR):
        print(f"❌ 错误: 源目录不存在: {SCRIPTS_DIR}")
        return False
    
    # 检查是否有DOCX文件
    docx_files = [f for f in os.listdir(SCRIPTS_DIR) if f.endswith('.docx')]
    if not docx_files:
        print(f"❌ 错误: 在 {SCRIPTS_DIR} 目录中未找到DOCX文件")
        return False
    
    print(f"✅ 找到 {len(docx_files)} 个DOCX文件")
    for file in docx_files:
        print(f"   - {file}")
    
    # 检查是否已经处理过
    if os.path.exists(SCRIPTS_FILE) and os.path.exists(MANUALS_FILE):
        print(f"\n✅ 检测到已处理的文本文件:")
        print(f"   - 剧本文件: {SCRIPTS_FILE} ({os.path.getsize(SCRIPTS_FILE) / 1024:.2f} KB)")
        print(f"   - 手册文件: {MANUALS_FILE} ({os.path.getsize(MANUALS_FILE) / 1024:.2f} KB)")
        return True
    
    # 如果没有处理过，提示运行处理脚本
    print(f"\n❌ 未检测到已处理的文本文件")
    print(f"   请先运行以下命令处理DOCX文件:")
    print(f"   cd agent-project")
    print(f"   python process_docx_scripts.py")
    return False


def analyze_text_file(file_path, description):
    """分析文本文件内容"""
    print(f"\n📊 分析{description}文件: {os.path.basename(file_path)}")
    print("=" * 50)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 基本统计
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.split('\n'))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        print(f"基本统计:")
        print(f"  - 字符数: {char_count:,}")
        print(f"  - 单词数: {word_count:,}")
        print(f"  - 行数: {line_count:,}")
        print(f"  - 段落数: {paragraph_count:,}")
        
        # 提取角色名（基于常见的剧本格式）
        # 角色名通常单独占一行，后面跟着对话
        potential_characters = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            # 简单的角色名检测：全大写、长度适中、后面有空行或对话
            if (line.isupper() and 2 <= len(line) <= 20 and 
                not any(c.isdigit() for c in line) and
                i+1 < len(lines) and lines[i+1].strip()):
                potential_characters.append(line)
        
        # 统计角色名出现次数
        character_counts = Counter(potential_characters)
        top_characters = character_counts.most_common(10)
        
        if top_characters:
            print(f"\n检测到的角色（前10名）:")
            for character, count in top_characters:
                print(f"  - {character}: {count} 次")
        
        # 提取关键词
        # 简单的中文关键词提取（排除常见停用词）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        common_stopwords = {'的', '了', '和', '是', '在', '有', '我', '他', '她', '它', '这', '那', '你', '们', '就', '都'}
        filtered_words = [w for w in chinese_words if w not in common_stopwords and len(w) >= 2]
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(15)
        
        if top_words:
            print(f"\n关键词（前15名）:")
            for word, count in top_words[:10]:
                print(f"  - {word}: {count} 次")
        
        # 显示前几个段落作为样本
        print(f"\n样本内容预览:")
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            sample_paragraph = paragraphs[0][:200] + ('...' if len(paragraphs[0]) > 200 else '')
            print(f"{sample_paragraph}")
        
        return True
    
    except Exception as e:
        print(f"❌ 分析文件时出错: {str(e)}")
        return False


def create_local_training_index():
    """创建本地训练索引文件"""
    print(f"\n🔄 创建本地训练索引...")
    
    try:
        # 读取剧本和手册内容
        with open(SCRIPTS_FILE, 'r', encoding='utf-8') as f:
            scripts_text = f.read()
        
        with open(MANUALS_FILE, 'r', encoding='utf-8') as f:
            manuals_text = f.read()
        
        # 创建简单的索引文件
        index_content = f"""# 剧本训练数据索引

## 基本信息
- 源目录: {SCRIPTS_DIR}
- 剧本文件大小: {os.path.getsize(SCRIPTS_FILE) / 1024:.2f} KB
- 手册文件大小: {os.path.getsize(MANUALS_FILE) / 1024:.2f} KB
- 索引创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 数据统计
- 剧本字符数: {len(scripts_text):,}
- 剧本单词数: {len(scripts_text.split()):,}
- 手册字符数: {len(manuals_text):,}
- 手册单词数: {len(manuals_text.split()):,}

## 使用说明
1. 这些文本数据已准备好用于训练
2. 您可以直接使用这些数据或进一步处理
3. 当Docker服务可用时，可以导入到完整系统中
"""
        
        index_file = os.path.join(OUTPUT_DIR, "training_index.md")
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        print(f"✅ 本地训练索引已创建: {index_file}")
        return True
    
    except Exception as e:
        print(f"❌ 创建索引时出错: {str(e)}")
        return False


def show_usage_guide():
    """显示使用指南"""
    print("\n📖 使用指南")
    print("=" * 50)
    print("训练完成后，您可以:")
    print("1. 查看分析结果和统计数据")
    print("2. 直接使用output目录中的文本文件进行进一步处理")
    print("3. 当网络环境改善时，重新启动Docker服务进行完整训练")
    print("\n可用的文本文件:")
    print(f"   - {SCRIPTS_FILE}")
    print(f"   - {MANUALS_FILE}")
    print(f"   - {os.path.join(OUTPUT_DIR, 'training_index.md')}")
    print("=" * 50)


def main():
    """主函数"""
    print("🚀 开始使用剧本数据进行简化训练")
    print(f"📂 源文件夹: {SCRIPTS_DIR}")
    print(f"📂 输出文件夹: {OUTPUT_DIR}")
    
    # 检查并处理DOCX文件
    if not check_and_process_docx_files():
        return
    
    # 分析剧本文件
    analyze_text_file(SCRIPTS_FILE, "剧本")
    
    # 分析手册文件
    analyze_text_file(MANUALS_FILE, "手册")
    
    # 创建本地训练索引
    create_local_training_index()
    
    # 显示使用指南
    show_usage_guide()
    
    print("\n🎉 简化训练完成！您现在可以使用生成的文本文件进行进一步分析或训练。")


if __name__ == "__main__":
    import time
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  操作被用户中断")
    except Exception as e:
        print(f"\n❌ 运行时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 脚本执行结束")