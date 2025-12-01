#!/usr/bin/env python3
"""
简化版Word文档信息提取器
不依赖外部库，提供基本信息和手动处理指导
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import base64

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleDocxInfo:
    """简化版文档信息提取器"""

    def __init__(self, docx_path: str):
        self.docx_path = Path(docx_path)
        self.file_info = {}
        self.is_readable = self._check_if_readable()

    def _check_if_readable(self) -> bool:
        """检查文件是否可读"""
        try:
            # 检查文件是否存在
            if not self.docx_path.exists():
                return False

            # 检查文件大小
            file_size = self.docx_path.stat().st_size
            if file_size == 0:
                return False

            # 检查是否是docx文件
            if self.docx_path.suffix.lower() != '.docx':
                return False

            return True

        except Exception as e:
            logger.error(f"检查文件失败: {e}")
            return False

    def extract_basic_info(self) -> dict:
        """提取基本信息"""
        if not self.is_readable:
            return {"error": "文件不可读"}

        try:
            # 基本文件信息
            stat = self.docx_path.stat()

            self.file_info = {
                "filename": self.docx_path.name,
                "file_path": str(self.docx_path),
                "file_size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }

            # 从文件名推断角色
            filename = self.docx_path.name
            if "柯太太" in filename:
                self.file_info["role"] = "柯太太"
                self.file_info["role_type"] = "character"
            elif "柯少爷" in filename:
                self.file_info["role"] = "柯少爷"
                self.file_info["role_type"] = "character"
            elif "云晴" in filename:
                self.file_info["role"] = "云晴"
                self.file_info["role_type"] = "character"
            elif "零四" in filename:
                self.file_info["role"] = "零四"
                self.file_info["role_type"] = "character"
            elif "雾晓" in filename:
                self.file_info["role"] = "雾晓"
                self.file_info["role_type"] = "character"
            elif "手册" in filename:
                self.file_info["role"] = "游戏手册"
                self.file_info["role_type"] = "manual"
            elif "线索" in filename:
                self.file_info["role"] = "线索材料"
                self.file_info["role_type"] = "clues"
            else:
                self.file_info["role"] = "未知"
                self.file_info["role_type"] = "unknown"

            return self.file_info

        except Exception as e:
            return {"error": str(e)}

    def extract_images_info(self) -> dict:
        """提取图片信息"""
        if not self.is_readable:
            return {"error": "文件不可读"}

        try:
            images_info = []

            # 使用zipfile检查docx结构
            import zipfile
            with zipfile.ZipFile(self.docx_path, 'r') as zip_file:
                # 查找图片文件
                for file in zip_file.filelist:
                    if file.filename.startswith('word/media/') and \
                       file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):

                        # 提取图片数据
                        image_data = zip_file.read(file.filename)
                        image_info = {
                            "filename": os.path.basename(file.filename),
                            "internal_path": file.filename,
                            "size_bytes": len(image_data),
                            "description": self._generate_image_description(file.filename),
                            "extracted": True
                        }

                        # 转换为base64（可选，用于保存图片信息）
                        try:
                            image_info["base64"] = base64.b64encode(image_data).decode('utf-8')
                        except:
                            image_info["base64"] = ""

                        images_info.append(image_info)

            return {
                "image_count": len(images_info),
                "images": images_info
            }

        except Exception as e:
            logger.error(f"提取图片信息失败: {e}")
            return {"error": str(e), "image_count": 0}

    def _generate_image_description(self, filename: str) -> str:
        """生成图片描述"""
        base_name = os.path.basename(filename)

        if "柯太太" in base_name:
            return "柯太太的照片/证件照"
        elif "柯少爷" in base_name:
            return "柯少爷的照片/证件照"
        elif "云晴" in base_name:
            return "云晴的照片/证件照"
        elif "零四" in base_name:
            return "零四的照片/证件照"
        elif "雾晓" in base_name:
            return "雾晓的照片/证件照"
        elif "手册" in base_name:
            return "游戏手册中的图片/说明图"
        elif "线索" in base_name:
            return f"线索图片 - {base_name}"
        elif "时间线" in base_name:
            return "时间线/时间轴图片"
        elif "地图" in base_name or "平面图" in base_name:
            return "地图/平面图"
        else:
            return f"剧情相关图片 - {base_name}"

    def get_manual_conversion_guide(self) -> str:
        """获取手动转换指导"""
        return f"""
手动转换指导 - {self.docx_path.name}

步骤1: 提取文本内容
- 打开Word文档
- 全选文本 (Ctrl+A)
- 复制 (Ctrl+C)
- 粘贴到文本文件中

步骤2: 提取图片内容
- 右键点击图片 → 另存为
- 保存为 PNG 或 JPG 格式
- 记录图片对应的文字说明

步骤3: 提取关键信息
- 确定角色姓名
- 识别关键线索
- 标记时间线
- 记录重要对话

建议输出格式:
角色: {self.file_info.get("role", "未知")}
文本内容: [粘贴的文本]
图片: [图片描述和文件名]
关键线索: [列出关键线索]
时间线: [重要时间点]
"""


def analyze_all_files(docx_files):
    """分析所有docx文件"""
    print("\n" + "="*60)
    print("Word文档分析报告")
    print("="*60)

    file_analysis = []
    character_files = {}
    manual_files = []
    clues_files = []

    for docx_file in docx_files:
        if not os.path.exists(docx_file):
            print(f"❌ 文件不存在: {docx_file}")
            continue

        print(f"\n📄 分析文件: {os.path.basename(docx_file)}")
        analyzer = SimpleDocxInfo(docx_file)

        # 基本信息
        basic_info = analyzer.extract_basic_info()
        if "error" in basic_info:
            print(f"❌ 错误: {basic_info['error']}")
            continue

        print(f"   角色: {basic_info['role']}")
        print(f"   类型: {basic_info['role_type']}")
        print(f"   大小: {basic_info['file_size']} bytes")

        # 图片信息
        images_info = analyzer.extract_images_info()
        if "error" not in images_info:
            print(f"   图片: {images_info['image_count']} 张")
            for img in images_info['images'][:3]:  # 只显示前3张
                print(f"      - {img['description']}")
        else:
            print(f"   图片: 检查失败")

        # 分类存储
        analysis_data = {
            "file_info": basic_info,
            "images_info": images_info
        }

        if basic_info['role_type'] == 'character':
            character_files[basic_info['role']] = analysis_data
        elif basic_info['role_type'] == 'manual':
            manual_files.append(analysis_data)
        elif basic_info['role_type'] == 'clues':
            clues_files.append(analysis_data)

        file_analysis.append(analysis_data)

    # 输出总结
    print("\n" + "="*60)
    print("分析总结")
    print("="*60)
    print(f"总文件数: {len(file_analysis)}")
    print(f"角色文件: {len(character_files)}")
    print(f"手册文件: {len(manual_files)}")
    print(f"线索文件: {len(clues_files)}")

    print("\n角色列表:")
    for role in character_files.keys():
        print(f"  - {role}")

    # 生成处理建议
    print("\n📋 数据处理建议:")
    print("1. 将所有角色剧本合并成一个完整文本")
    print("2. 提取所有关键图片并标注说明")
    print("3. 整理时间线和关键线索")
    print("4. 创建标准训练数据格式")

    return {
        "all_files": file_analysis,
        "characters": character_files,
        "manuals": manual_files,
        "clues": clues_files
    }


def create_training_sample_template():
    """创建训练数据模板"""
    template = {
        "script_id": "柯家庄园谋杀案_001",
        "title": "柯家庄园谋杀案",
        "original_script": "这里放入完整剧本文本...",
        "compressed_script": "这里放入压缩后的剧本...",
        "compression_ratio": 0.6,
        "compression_level": "medium",
        "logic_integrity": 0.9,
        "story_coherence": 0.85,
        "playability_score": 0.88,
        "preserved_elements": [
            "柯太太的秘密",
            "零四的真实身份",
            "凶案时间线",
            "关键证据照片"
        ],
        "key_images": [
            {
                "description": "案发现场照片",
                "filename": "scene_photo_01.jpg",
                "importance": "high"
            }
        ],
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "version": "v1.0",
            "analyzer": "manual"
        }
    }

    # 保存模板
    output_dir = Path("/home/ubt/桌面/agent-project/data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    template_file = output_dir / "training_sample_template.json"
    with open(template_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"\n📝 训练数据模板已创建: {template_file}")
    print("请根据模板格式填充您的剧本数据")


def main():
    """主函数"""
    docx_files = [
        "/home/ubt/桌面/agent-project/01 绷带女人 柯太太_QQ浏览器转格式.docx",
        "/home/ubt/桌面/agent-project/02 年轻男子 柯少爷_QQ浏览器转格式.docx",
        "/home/ubt/桌面/agent-project/03 女仆 云晴_QQ浏览器转格式.docx",
        "/home/ubt/桌面/agent-project/04 胡茬男人 零四_QQ浏览器转格式 (1).docx",
        "/home/ubt/桌面/agent-project/05 洋裙女子 雾晓_QQ浏览器转格式.docx",
        "/home/ubt/桌面/agent-project/手册_QQ浏览器转格式.docx",
        "/home/ubt/桌面/agent-project/线索_QQ浏览器转格式.docx"
    ]

    # 分析文件
    analysis_result = analyze_all_files(docx_files)

    # 创建训练数据模板
    create_training_sample_template()

    print(f"\n🎯 下一步操作建议:")
    print(f"1. 手动提取每个Word文件的文本内容")
    print(f"2. 提取并保存所有关键图片")
    print(f"3. 使用训练数据模板创建标准格式")
    print(f"4. 运行: python scripts/create_training_data.py")


if __name__ == "__main__":
    main()