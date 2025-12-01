#!/usr/bin/env python3
"""
专用压缩模型训练脚本
"""

import os
import sys
import argparse
import yaml
import logging
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models.specialized_compression_model import SpecializedCompressionModel
from core.data.data_pipeline import DataPipeline
from core.training.trainer import CompressionTrainer, TrainingConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="训练专用压缩模型")
    parser.add_argument("--config", type=str, required=True, help="训练配置文件路径")
    parser.add_argument("--data-dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="输出目录")
    parser.add_argument("--resume", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--eval-only", action="store_true", help="仅进行评估")
    parser.add_argument("--device", type=str, default="auto", help="训练设备 (auto/cpu/cuda)")

    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config(args.config)

        # 设备配置
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        logger.info(f"使用设备: {device}")

        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志目录
        os.makedirs("logs", exist_ok=True)

        # 初始化数据管道
        logger.info("初始化数据管道...")
        data_pipeline = DataPipeline(
            data_dir=args.data_dir,
            tokenizer_name=config.get('model_name', 't5-large'),
            max_length=config.get('max_length', 2048),
            batch_size=config.get('batch_size', 4),
            num_workers=config.get('num_workers', 4)
        )

        # 准备训练数据
        if not args.eval_only:
            logger.info("准备训练数据...")
            raw_data_sources = config.get('data_sources', [])
            if not raw_data_sources:
                # 使用默认示例数据
                sample_data_path = Path(args.data_dir) / "samples" / "sample_data.json"
                if sample_data_path.exists():
                    raw_data_sources = [str(sample_data_path)]
                else:
                    # 创建示例数据
                    logger.info("创建示例训练数据...")
                    data_pipeline.create_sample_data(str(sample_data_path), num_samples=100)
                    raw_data_sources = [str(sample_data_path)]

            # 数据处理
            data_files = data_pipeline.prepare_training_data(
                raw_data_sources=raw_data_sources,
                output_dir=str(output_dir / "processed_data"),
                apply_augmentation=config.get('apply_augmentation', True),
                augmentation_factor=config.get('augmentation_factor', 2.0)
            )

            # 创建数据加载器
            data_loaders = data_pipeline.create_data_loaders(data_files)

            train_loader = data_loaders.get("train")
            val_loader = data_loaders.get("val")
            test_loader = data_loaders.get("test")

            if not train_loader:
                raise ValueError("训练数据加载失败")

            logger.info(f"训练数据: {len(train_loader.dataset)} 样本")
            if val_loader:
                logger.info(f"验证数据: {len(val_loader.dataset)} 样本")
            if test_loader:
                logger.info(f"测试数据: {len(test_loader.dataset)} 样本")

        # 初始化模型
        logger.info("初始化专用压缩模型...")
        model = SpecializedCompressionModel(
            model_name=config.get('model_name', 't5-large'),
            hidden_size=config.get('hidden_size', 768),
            vocab_size=config.get('vocab_size', 32128)
        )

        # 初始化训练器
        logger.info("初始化训练器...")
        trainer = CompressionTrainer(
            model=model,
            data_pipeline=data_pipeline,
            config=config
        )

        # 恢复训练
        if args.resume:
            logger.info(f"从检查点恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # 仅评估模式
        if args.eval_only:
            logger.info("运行模型评估...")
            if test_loader:
                eval_results = trainer.evaluate_model(test_loader)
                logger.info("评估结果:")
                for metric, value in eval_results.items():
                    logger.info(f"  {metric}: {value}")
            else:
                logger.error("没有测试数据可用于评估")
            return

        # 开始训练
        logger.info("开始训练...")
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )

        # 训练完成后的评估
        if test_loader:
            logger.info("训练完成，进行最终评估...")
            final_eval_results = trainer.evaluate_model(test_loader)

            logger.info("最终评估结果:")
            for metric, value in final_eval_results.items():
                logger.info(f"  {metric}: {value}")

            # 保存评估结果
            eval_results_path = output_dir / "evaluation_results.json"
            import json
            with open(eval_results_path, 'w', encoding='utf-8') as f:
                json.dump(final_eval_results, f, indent=2, ensure_ascii=False)

            logger.info(f"评估结果已保存: {eval_results_path}")

        logger.info("训练完成！")
        logger.info(f"最佳验证损失: {trainer.best_val_loss}")
        logger.info(f"输出目录: {output_dir}")

    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()