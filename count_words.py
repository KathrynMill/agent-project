#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计原始Word文档和生成的文本文件的字数
计算压缩率和内容保留情况
"""

import os
import sys
import re
from docx import Document

def count_words_in_docx(file_path):
    """统计Word文档中的字数"""
    try:
        doc = Document(file_path)
        total_words = 0
        for paragraph in doc.paragraphs:
            # 清理文本并统计字数
            text = paragraph.text.strip()
            if text:
                # 中文按字符算，英文按空格分割算单词
                # 这里简化处理，按字符数计算
                total_words += len(text)
        return total_words
    except Exception as e:
        print(f"⚠️  统计文件 {file_path} 字数失败: {e}")
        return 0

def count_words_in_txt(file_path):
    """统计文本文件中的字数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 清理文本（移除空行和多余空格）
            content = re.sub(r'\s+', '', content)
            return len(content)
    except Exception as e:
        print(f"⚠️  统计文件 {file_path} 字数失败: {e}")
        return 0

def main():
    """主函数"""
    print("📊 开始统计文档字数...\n")
    
    # 原始文档目录
    original_dirs = [
        r"c:\Users\11928\Desktop\linshi\如是我观 - 副本\如是我观 - 副本",
        r"c:\Users\11928\Desktop\linshi\因火成烟 - 副本\因火成烟 - 副本"
    ]
    
    # 生成的文件目录
    output_dir = r"c:\Users\11928\Desktop\linshi\output"
    
    # 统计原始文档
    original_total = 0
    for dir_path in original_dirs:
        print(f"📂 目录: {os.path.basename(dir_path)}")
        dir_total = 0
        for filename in os.listdir(dir_path):
            if filename.lower().endswith('.docx') and not filename.startswith('~$'):
                file_path = os.path.join(dir_path, filename)
                words = count_words_in_docx(file_path)
                dir_total += words
                print(f"  - {filename}: {words:,} 字")
        original_total += dir_total
        print(f"  目录总计: {dir_total:,} 字\n")
    
    print(f"📝 原始文档总字数: {original_total:,} 字\n")
    
    # 统计生成的文件
    output_total = 0
    print("📂 生成的文件统计:")
    for filename in os.listdir(output_dir):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(output_dir, filename)
            words = count_words_in_txt(file_path)
            output_total += words
            print(f"  - {filename}: {words:,} 字")
    
    print(f"  生成文件总字数: {output_total:,} 字\n")
    
    # 计算压缩率
    compression_rate = ((original_total - output_total) / original_total * 100) if original_total > 0 else 0
    retention_rate = (output_total / original_total * 100) if original_total > 0 else 0
    
    print("📊 统计结果汇总:")
    print(f"原始文档总字数: {original_total:,} 字")
    print(f"生成文件总字数: {output_total:,} 字")
    print(f"文件压缩率: {compression_rate:.2f}%")
    print(f"内容保留率: {retention_rate:.2f}%")
    
    # 分析结果
    if retention_rate > 95:
        print("💡 分析: 当前基本是原文转换，几乎没有压缩。")
    elif retention_rate > 80:
        print("💡 分析: 有轻微压缩，但主要保留了原文内容。")
    elif retention_rate > 50:
        print("💡 分析: 有一定压缩，保留了大部分核心内容。")
    else:
        print("💡 分析: 压缩程度较高，可能需要检查是否保留了关键信息。")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())