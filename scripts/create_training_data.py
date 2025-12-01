#!/usr/bin/env python3
"""
创建训练数据 - 手动数据处理工具
基于柯家庄园谋杀案的Word文档
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ScriptDataBuilder:
    """剧本数据构建器"""

    def __init__(self):
        self.script_data = {
            "script_id": "ke_mansion_murder_001",
            "title": "柯家庄园谋杀案",
            "version": "v1.0",
            "created_at": datetime.now().isoformat()
        }

    def set_character_scripts(self, characters_data):
        """设置角色剧本"""
        self.script_data["character_scripts"] = characters_data
        logger.info(f"设置了 {len(characters_data)} 个角色剧本")

    def set_game_manual(self, manual_data):
        """设置游戏手册"""
        self.script_data["game_manual"] = manual_data
        logger.info("设置了游戏手册")

    def set_clues_data(self, clues_data):
        """设置线索数据"""
        self.script_data["clues"] = clues_data
        logger.info("设置了线索数据")

    def create_full_script(self):
        """创建完整剧本文本"""
        full_text_parts = []

        # 标题和概述
        full_text_parts.append("剧本标题：柯家庄园谋杀案")
        full_text_parts.append("\n剧本概述：这是一个发生在柯家庄园的复杂谋杀案，涉及多个角色和隐藏的秘密。")

        # 角色介绍
        full_text_parts.append("\n角色介绍：")
        characters = self.script_data.get("character_scripts", {})
        for role, data in characters.items():
            full_text_parts.append(f"\n{role}：")
            if isinstance(data, dict) and "text" in data:
                full_text_parts.append(data["text"])
            elif isinstance(data, str):
                full_text_parts.append(data)

        # 游戏手册
        manual = self.script_data.get("game_manual", {})
        if manual:
            full_text_parts.append("\n\n游戏手册：")
            if isinstance(manual, dict) and "text" in manual:
                full_text_parts.append(manual["text"])
            elif isinstance(manual, str):
                full_text_parts.append(manual)

        # 线索材料
        clues = self.script_data.get("clues", {})
        if clues:
            full_text_parts.append("\n\n线索材料：")
            if isinstance(clues, dict) and "text" in clues:
                full_text_parts.append(clues["text"])
            elif isinstance(clues, str):
                full_text_parts.append(clues)

        self.script_data["full_text"] = "\n".join(full_text_parts)
        logger.info(f"创建了完整剧本，长度: {len(self.script_data['full_text'])} 字符")

    def extract_key_elements(self):
        """提取关键元素"""
        elements = {
            "characters": list(self.script_data.get("character_scripts", {}).keys()),
            "game_type": "剧本杀",
            "complexity": "medium",
            "estimated_duration_hours": 3.5,
            "difficulty_level": "medium",
            "player_count": len(self.script_data.get("character_scripts", {}))
        }

        # 从内容中推断关键线索（简化版）
        full_text = self.script_data.get("full_text", "")

        # 简单的关键词匹配
        key_elements = []
        if "柯太太" in full_text:
            key_elements.append("柯太太的秘密身份")
        if "零四" in full_text:
            key_elements.append("零四的真实身份")
        if "死亡" in full_text:
            key_elements.append("死亡案件")
        if "照片" in full_text or "图片" in full_text:
            key_elements.append("关键照片证据")
        if "时间" in full_text:
            key_elements.append("重要时间点")

        elements["key_elements"] = key_elements
        return elements

    def create_training_samples(self, compression_ratios=[0.5, 0.6, 0.7]):
        """创建多个压缩比例的训练样本"""
        samples = []

        key_elements = self.extract_key_elements()
        full_text = self.script_data.get("full_text", "")

        for ratio in compression_ratios:
            sample = {
                "script_id": f"{self.script_data['script_id']}_{ratio}",
                "title": self.script_data["title"],
                "original_script": full_text,
                "compression_ratio": ratio,
                "compression_level": self._get_compression_level(ratio),
                "key_elements": key_elements["key_elements"],
                "metadata": {
                    "character_count": key_elements["player_count"],
                    "complexity": key_elements["complexity"],
                    "difficulty": key_elements["difficulty_level"],
                    "created_at": datetime.now().isoformat(),
                    "compression_strategy": "balanced"
                }
            }
            samples.append(sample)

        return samples

    def _get_compression_level(self, ratio):
        """根据压缩比例确定压缩级别"""
        if ratio >= 0.7:
            return "light"
        elif ratio >= 0.5:
            return "medium"
        else:
            return "heavy"

    def save_to_json(self, output_path: str):
        """保存到JSON文件"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 创建数据结构
        output_data = {
            "script_info": {
                "script_id": self.script_data["script_id"],
                "title": self.script_data["title"],
                "created_at": self.script_data["created_at"],
                "character_count": len(self.script_data.get("character_scripts", {}))
            },
            "character_scripts": self.script_data.get("character_scripts", {}),
            "game_manual": self.script_data.get("game_manual", {}),
            "clues": self.script_data.get("clues", {}),
            "full_text": self.script_data.get("full_text", ""),
            "key_elements": self.extract_key_elements(),
            "training_samples": self.create_training_samples()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"数据已保存到: {output_file}")
        return output_file


def create_sample_data():
    """创建示例数据（用于演示）"""
    builder = ScriptDataBuilder()

    # 设置角色剧本（简化版示例）
    character_scripts = {
        "柯太太": """柯太太，45岁，富商张三的妻子
性格：精明强干，善于观察，但隐藏着秘密
背景：来自普通家庭，嫁入豪门后生活优渥
秘密：实际上并不爱张三，有自己的情人
动机：为了财产和自由""",

        "柯少爷": """柯少爷，25岁，张三和柯太太的儿子
性格：叛逆任性，不喜欢家庭的束缚
背景：在国外留学归来，思想西化
秘密：知道父亲的商业秘密
动机：争夺家族财产""",

        "云晴": """云晴，28岁，女仆
性格：温柔体贴，工作认真负责
背景：来自农村，在柯家工作多年
秘密：与柯少爷有私情
动机：保护爱情和地位""",

        "零四": """零四，50岁，神秘的访客
性格：沉默寡言，举止可疑
背景：自称是张三的生意伙伴
秘密：实际上是来调查真相的侦探
动机：为父亲报仇""",

        "雾晓": """雾晓，26岁，张三的秘书
性格：干练高效，深得信任
背景：名牌大学毕业，能力出众
秘密：被某势力收买来监视柯家
动机：完成雇主的任务"""
    }

    # 设置游戏手册（简化版）
    game_manual = """柯家庄园谋杀案游戏手册

游戏背景：
时间：现代都市
地点：柯家庄园
玩家人数：5人
游戏时长：3-4小时

游戏规则：
1. 每位玩家选择一个角色
2. 阅读个人剧本
3. 调查案件真相
4. 在规定时间内找出凶手

胜利条件：
- 找出真正的凶手
- 提供充分证据
- 保护自己不被怀疑

线索分布：
- 现场线索：照片、物品、痕迹
- 人物线索：对话、行为、时间
- 隐藏线索：秘密文件、录音"""

    # 设置线索材料（简化版）
    clues = """案件线索汇总

关键时间线：
18:00 - 晚餐开始
19:30 - 发现尸体
20:00 - 警察到达
21:00 - 调查开始

关键线索：
1. 死者：张三，死于毒药
2. 死亡时间：19:00-19:30之间
3. 死因：急性中毒
4. 嫌疑人：所有家庭成员

物理证据：
- 带有指纹的毒药瓶
- 张三的遗书
- 闭路电视录像
- 手机通话记录

心理证据：
- 家庭矛盾
- 财产纠纷
- 感情关系
- 职业竞争"""

    builder.set_character_scripts(character_scripts)
    builder.set_game_manual(game_manual)
    builder.set_clues_data(clues)

    # 创建完整剧本
    builder.create_full_script()

    # 保存数据
    output_path = "/home/ubt/桌面/agent-project/data/samples/柯家庄园谋杀案.json"
    saved_file = builder.save_to_json(output_path)

    print(f"\n🎭 示例剧本数据已创建: {saved_file}")
    print(f"📝 角色数量: {len(character_scripts)}")
    print(f"📖 包含手册: {game_manual is not None}")
    print(f"🔍 包含线索: {clues is not None}")
    print(f"📄 文本长度: {len(builder.script_data['full_text'])} 字符")


def create_manual_processing_guide():
    """创建手动处理指导"""
    guide = """
# 柯家庄园谋杀案 - 手动数据处理指导

## 📋 步骤1: 提取文本内容

对每个角色文件：
1. 打开对应的Word文档
2. 复制所有文本内容
3. 粘贴到下面对应位置

角色对应文件：
- 01 绷带女人 柯太太.docx → 柯太太剧本
- 02 年轻男子 柯少爷.docx → 柯少爷剧本
- 03 女仆 云晴.docx → 云晴剧本
- 04 胡茬男人 零四.docx → 零四剧本
- 05 洋裙女子 雾晓.docx → 雾晓剧本

## 🖼️ 步骤2: 提取图片内容

对每个文档：
1. 逐个右键点击图片
2. 选择"另存为图片"
3. 保存为PNG格式，命名规则：
   - 角色名_描述_序号.png
   - 例如：柯太太_证件照_01.png

## 📝 步骤3: 识别关键信息

请为每个剧本标记：
- 角色的基本信息
- 关键对话内容
- 隐藏的秘密
- 动机和目的

## 🎯 步骤4: 整理线索

从手册和线索文件中提取：
- 案件基本信息
- 时间线
- 物理证据
- 调查方向

## 📁 步骤5: 填充模板

将提取的信息填入训练数据模板
"""

    guide_file = Path("/home/ubt/桌面/agent-project/data/samples/MANUAL_PROCESSING_GUIDE.md")
    guide_file.parent.mkdir(parents=True, exist_ok=True)

    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)

    print(f"📖 手动处理指导已创建: {guide_file}")


def main():
    """主函数"""
    print("🎭 柯家庄园谋杀案 - 训练数据处理")
    print("="*50)

    # 创建示例数据
    create_sample_data()

    # 创建手动处理指导
    create_manual_processing_guide()

    print("\n" + "="*50)
    print("📋 处理流程总结")
    print("="*50)
    print("1. ✅ 已创建示例训练数据")
    print("2. 📖 已创建手动处理指导")
    print("3. 🎯 下一步: 手动提取真实数据")
    print("4. 📝 填充训练数据模板")
    print("5. 🚀 开始模型训练")

    print(f"\n💡 提示:")
    print("- 建议先使用示例数据进行测试")
    print("- 然后手动提取真实数据")
    print("- 质量比数量更重要")
    print("- 包含图片的训练样本效果更好")


if __name__ == "__main__":
    main()