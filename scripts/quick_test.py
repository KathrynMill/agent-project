#!/usr/bin/env python3
"""
专用压缩模型快速测试脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models.specialized_compression_model import (
    SpecializedCompressionModel, CompressionConfig, CompressionLevel
)
from core.services.compression_service import CompressionService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_basic():
    """测试模型基本功能"""
    logger.info("开始基本功能测试...")

    try:
        # 创建模型
        model = SpecializedCompressionModel(
            model_name='t5-large',
            hidden_size=768,
            vocab_size=32128
        )

        # 测试文本
        test_script = """
        剧本标题：神秘的 mansion

        人物介绍：
        张三：富有的商人，45岁，性格多疑
        李四：私人医生，38岁，看起来很紧张
        王五：律师，50岁，言辞严谨
        赵六：管家，60岁，在这个家里工作了20年

        剧本正文：
        第一幕：发现尸体
        张三："这...这是怎么回事？李医生，你快来看看！"
        李四：（检查尸体）"他已经死了至少两个小时了。"
        王五："我们需要报警。"
        赵六："老爷，让我来处理吧。"

        第二幕：调查开始
        张三："各位，我们都有嫌疑。让我先问问赵管家。"
        赵六："我一直在这里工作，从没见过这种事。"
        李四："死因是中毒，但毒药很特别。"
        王五："我们需要检查每个人的不在场证明。"

        第三幕：真相大白
        张三："其实我知道是谁干的。"
        李四："是谁？"
        张三："就是你，李医生！你给他下了慢性毒药。"
        李四："你胡说！我有证据证明我的清白。"
        王五："让我来看看证据。"

        第四幕：最终揭秘
        王五："证据显示，真正的凶手是赵管家。"
        赵六："为什么要这么做？"
        王五："因为你贪图主人的财产。"
        赵六："我认罪。"

        线索：
        1. 受害者死前喝了管家泡的茶
        2. 医生发现了特殊的毒药成分
        3. 律师找到了遗嘱的变更记录
        4. 管家有经济困难

        时间线：
        14:00 - 受害者喝茶
        15:30 - 发现尸体
        16:00 - 警察到达
        17:00 - 侦破案件
        """

        # 创建压缩配置
        config = CompressionConfig(
            target_ratio=0.6,  # 压缩到60%
            compression_level=CompressionLevel.MEDIUM
        )

        logger.info("执行压缩...")
        result = model(test_script, config)

        logger.info("压缩结果:")
        logger.info(f"  原始长度: {len(test_script)} 字符")
        logger.info(f"  压缩后长度: {len(result['compressed_text'])} 字符")
        logger.info(f"  实际压缩比例: {len(result['compressed_text']) / len(test_script):.3f}")

        metrics = result['metrics']
        logger.info(f"  逻辑完整性: {metrics.logic_integrity:.3f}")
        logger.info(f"  故事连贯性: {metrics.story_coherence:.3f}")
        logger.info(f"  可玩性分数: {metrics.playability_score:.3f}")
        logger.info(f"  长度准确性: {metrics.length_accuracy:.3f}")
        logger.info(f"  综合质量: {metrics.overall_quality:.3f}")

        logger.info("压缩后的剧本:")
        print(result['compressed_text'])

        return True

    except Exception as e:
        logger.error(f"基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compression_service():
    """测试压缩服务"""
    logger.info("开始压缩服务测试...")

    try:
        # 创建压缩服务
        service = CompressionService()

        # 测试文本
        test_script = """
        这是一个测试剧本，包含了基本的故事情节、人物对话和线索。
        张三说："我发现了一个重要的线索。"
        李四回答："是什么线索？"
        张三："现场的脚印很不寻常。"
        王五突然出现："你们在讨论什么？"
        线索1：奇怪的脚印
        线索2：失踪的物品
        时间线：早上8点-晚上10点
        """

        # 测试估算
        logger.info("测试压缩估算...")
        estimation = service.estimate_compression(test_script, 2.0)

        logger.info("估算结果:")
        for key, value in estimation.items():
            logger.info(f"  {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"压缩服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_strategies():
    """测试不同压缩策略"""
    logger.info("开始策略测试...")

    try:
        # 创建压缩服务
        service = CompressionService()

        test_script = """
        复杂的剧本内容，包含多个角色、线索和情节转折。
        主角：张三，李四，王五，赵六，钱七
        地点：豪宅，花园，书房，密室
        时间：三天三夜
        线索：信件，钥匙，照片，录音，日记
        情节：谋杀案，调查，误导，真相，复仇
        """ * 5  # 重复5次增加长度

        strategies = ["balanced", "preserve_logic", "preserve_story", "fast"]
        target_hours = 2.0

        results = {}

        for strategy in strategies:
            logger.info(f"测试策略: {strategy}")

            estimation = service.estimate_compression(test_script, target_hours, strategy)
            results[strategy] = estimation

            logger.info(f"  预计压缩比例: {estimation.get('target_compression_ratio', 'N/A')}")
            logger.info(f"  预计质量分数: {estimation.get('estimated_quality_score', 'N/A')}")
            logger.info(f"  可行性: {estimation.get('feasibility', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """主测试函数"""
    logger.info("=" * 50)
    logger.info("专用压缩模型快速测试")
    logger.info("=" * 50)

    # 测试基本功能
    basic_test_passed = test_model_basic()
    logger.info(f"基本功能测试: {'通过' if basic_test_passed else '失败'}")

    # 测试压缩服务
    service_test_passed = test_compression_service()
    logger.info(f"压缩服务测试: {'通过' if service_test_passed else '失败'}")

    # 测试不同策略
    strategy_results = test_different_strategies()
    logger.info(f"策略测试完成: 测试了 {len(strategy_results)} 种策略")

    logger.info("=" * 50)
    logger.info("快速测试完成")
    logger.info("=" * 50)

    if basic_test_passed and service_test_passed:
        logger.info("✅ 所有核心功能测试通过")
    else:
        logger.info("❌ 部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()