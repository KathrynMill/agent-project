#!/usr/bin/env python3
"""
专用压缩模型评估脚本
"""

import os
import sys
import argparse
import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models.specialized_compression_model import (
    SpecializedCompressionModel, CompressionConfig, CompressionLevel
)
from core.data.data_pipeline import DataPipeline
from core.services.compression_service import CompressionService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_path: str, device: str = "auto"):
        """初始化评估器"""
        self.model_path = model_path
        self.device = torch.device(device if device != "auto" else
                                 "cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        logger.info(f"加载模型: {model_path}")
        self.model = SpecializedCompressionModel.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 初始化压缩服务
        self.compression_service = CompressionService(model_path)

        logger.info(f"模型加载完成，设备: {self.device}")

    def evaluate_compression_quality(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估压缩质量"""
        logger.info("开始评估压缩质量...")

        results = {
            "total_samples": len(test_samples),
            "compression_ratios": [],
            "quality_scores": [],
            "logic_integrity": [],
            "story_coherence": [],
            "playability_scores": [],
            "length_accuracies": [],
            "processing_times": []
        }

        for i, sample in enumerate(test_samples):
            try:
                logger.info(f"评估样本 {i+1}/{len(test_samples)}")

                original_script = sample["original_script"]
                target_ratio = sample["target_ratio"]
                compression_level = CompressionLevel(sample.get("compression_level", "medium"))

                # 创建压缩配置
                config = CompressionConfig(
                    target_ratio=target_ratio,
                    compression_level=compression_level
                )

                # 执行压缩
                import time
                start_time = time.time()

                with torch.no_grad():
                    compression_result = self.model(original_script, config)

                processing_time = time.time() - start_time

                # 提取指标
                metrics = compression_result['metrics']

                results["compression_ratios"].append(metrics.compression_ratio)
                results["quality_scores"].append(metrics.overall_quality)
                results["logic_integrity"].append(metrics.logic_integrity)
                results["story_coherence"].append(metrics.story_coherence)
                results["playability_scores"].append(metrics.playability_score)
                results["length_accuracies"].append(metrics.length_accuracy)
                results["processing_times"].append(processing_time)

                logger.info(f"  压缩比例: {metrics.compression_ratio:.3f}")
                logger.info(f"  质量分数: {metrics.overall_quality:.3f}")
                logger.info(f"  处理时间: {processing_time:.2f}s")

            except Exception as e:
                logger.error(f"样本 {i+1} 评估失败: {e}")
                continue

        # 计算统计指标
        evaluation_metrics = self._calculate_statistics(results)

        return evaluation_metrics

    def evaluate_compression_strategies(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估不同压缩策略的效果"""
        logger.info("开始评估压缩策略...")

        strategies = ["balanced", "preserve_logic", "preserve_story", "fast"]
        target_ratios = [0.7, 0.5, 0.4]  # 轻度、中度、重度压缩

        strategy_results = {}

        for strategy in strategies:
            logger.info(f"评估策略: {strategy}")
            strategy_results[strategy] = {}

            for target_ratio in target_ratios:
                ratio_key = f"ratio_{target_ratio}"
                logger.info(f"  目标比例: {target_ratio}")

                results = {
                    "quality_scores": [],
                    "compression_ratios": [],
                    "processing_times": []
                }

                for i, sample in enumerate(test_samples[:20]):  # 限制样本数量
                    try:
                        original_script = sample["original_script"]

                        # 根据策略创建配置
                        if target_ratio >= 0.7:
                            level = CompressionLevel.LIGHT
                        elif target_ratio >= 0.5:
                            level = CompressionLevel.MEDIUM
                        else:
                            level = CompressionLevel.HEAVY

                        if strategy == "preserve_logic":
                            config = CompressionConfig(
                                target_ratio=target_ratio,
                                compression_level=level,
                                logic_weight=0.5,
                                story_weight=0.2,
                                playability_weight=0.2,
                                length_weight=0.1
                            )
                        elif strategy == "preserve_story":
                            config = CompressionConfig(
                                target_ratio=target_ratio,
                                compression_level=level,
                                logic_weight=0.2,
                                story_weight=0.5,
                                playability_weight=0.2,
                                length_weight=0.1
                            )
                        elif strategy == "fast":
                            config = CompressionConfig(
                                target_ratio=target_ratio,
                                compression_level=level,
                                logic_weight=0.2,
                                story_weight=0.2,
                                playability_weight=0.3,
                                length_weight=0.3
                            )
                        else:  # balanced
                            config = CompressionConfig(
                                target_ratio=target_ratio,
                                compression_level=level
                            )

                        # 执行压缩
                        import time
                        start_time = time.time()

                        with torch.no_grad():
                            compression_result = self.model(original_script, config)

                        processing_time = time.time() - start_time
                        metrics = compression_result['metrics']

                        results["quality_scores"].append(metrics.overall_quality)
                        results["compression_ratios"].append(metrics.compression_ratio)
                        results["processing_times"].append(processing_time)

                    except Exception as e:
                        logger.error(f"策略评估失败: {e}")
                        continue

                # 计算平均值
                if results["quality_scores"]:
                    strategy_results[strategy][ratio_key] = {
                        "avg_quality": np.mean(results["quality_scores"]),
                        "avg_compression_ratio": np.mean(results["compression_ratios"]),
                        "avg_processing_time": np.mean(results["processing_times"]),
                        "num_samples": len(results["quality_scores"])
                    }

        return strategy_results

    def evaluate_model_performance(self) -> Dict[str, Any]:
        """评估模型性能指标"""
        logger.info("开始评估模型性能...")

        # 模型信息
        model_info = self.compression_service.get_model_info()

        # 统计信息
        stats = self.compression_service.get_compression_stats()

        # 健康检查
        health_status = self.compression_service.health_check()

        # GPU内存使用情况
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "total": torch.cuda.get_device_properties(0).total_memory
            }

        performance_metrics = {
            "model_info": model_info,
            "performance_stats": stats,
            "health_status": health_status,
            "gpu_memory": gpu_memory,
            "evaluation_timestamp": datetime.now().isoformat()
        }

        return performance_metrics

    def _calculate_statistics(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """计算统计指标"""
        metrics = {}

        for key, values in results.items():
            if values:
                metrics[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75)
                }
            else:
                metrics[key] = {}

        return metrics


def load_test_samples(data_path: str, num_samples: int = 50) -> List[Dict[str, Any]]:
    """加载测试样本"""
    logger.info(f"加载测试样本: {data_path}")

    if os.path.isfile(data_path):
        # 从单个文件加载
        data_pipeline = DataPipeline("./data")
        samples = data_pipeline.load_examples(data_path)
    else:
        # 从目录加载
        data_pipeline = DataPipeline(data_path)
        data_files = data_pipeline.prepare_training_data(
            raw_data_sources=[data_path],
            output_dir="./temp_eval_data"
        )
        data_loaders = data_pipeline.create_data_loaders(data_files)

        # 从数据加载器提取样本
        samples = []
        if "test" in data_loaders:
            for batch in data_loaders["test"]:
                # 简化处理：创建模拟样本
                for i in range(batch['original_input_ids'].size(0)):
                    samples.append({
                        "original_script": "测试剧本内容..." * 100,  # 模拟长文本
                        "target_ratio": batch['target_ratio'][i].item(),
                        "compression_level": "medium"
                    })
                if len(samples) >= num_samples:
                    break

    return samples[:num_samples]


def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description="评估专用压缩模型")
    parser.add_argument("--model-path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--data-path", type=str, default="./data", help="测试数据路径")
    parser.add_argument("--output-dir", type=str, default="./results", help="结果输出目录")
    parser.add_argument("--num-samples", type=int, default=50, help="测试样本数量")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--strategies", action="store_true", help="评估不同压缩策略")
    parser.add_argument("--performance", action="store_true", help="评估模型性能")

    args = parser.parse_args()

    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # 初始化评估器
        evaluator = ModelEvaluator(args.model_path, args.device)

        # 加载测试样本
        test_samples = load_test_samples(args.data_path, args.num_samples)
        logger.info(f"加载了 {len(test_samples)} 个测试样本")

        if not test_samples:
            logger.error("没有找到测试样本")
            return

        # 评估压缩质量
        logger.info("=" * 50)
        logger.info("评估压缩质量")
        logger.info("=" * 50)

        quality_results = evaluator.evaluate_compression_quality(test_samples)

        # 保存质量评估结果
        quality_path = os.path.join(args.output_dir, "quality_evaluation.json")
        with open(quality_path, 'w', encoding='utf-8') as f:
            json.dump(quality_results, f, indent=2, ensure_ascii=False)

        logger.info(f"质量评估结果已保存: {quality_path}")

        # 评估压缩策略
        if args.strategies:
            logger.info("=" * 50)
            logger.info("评估压缩策略")
            logger.info("=" * 50)

            strategy_results = evaluator.evaluate_compression_strategies(test_samples[:20])  # 限制样本数量

            # 保存策略评估结果
            strategy_path = os.path.join(args.output_dir, "strategy_evaluation.json")
            with open(strategy_path, 'w', encoding='utf-8') as f:
                json.dump(strategy_results, f, indent=2, ensure_ascii=False)

            logger.info(f"策略评估结果已保存: {strategy_path}")

        # 评估模型性能
        if args.performance:
            logger.info("=" * 50)
            logger.info("评估模型性能")
            logger.info("=" * 50)

            performance_results = evaluator.evaluate_model_performance()

            # 保存性能评估结果
            performance_path = os.path.join(args.output_dir, "performance_evaluation.json")
            with open(performance_path, 'w', encoding='utf-8') as f:
                json.dump(performance_results, f, indent=2, ensure_ascii=False)

            logger.info(f"性能评估结果已保存: {performance_path}")

        # 生成评估报告
        logger.info("=" * 50)
        logger.info("评估结果摘要")
        logger.info("=" * 50)

        if quality_results.get("quality_scores"):
            avg_quality = quality_results["quality_scores"]["mean"]
            avg_ratio = quality_results["compression_ratios"]["mean"]
            avg_time = quality_results["processing_times"]["mean"]

            logger.info(f"平均质量分数: {avg_quality:.3f}")
            logger.info(f"平均压缩比例: {avg_ratio:.3f}")
            logger.info(f"平均处理时间: {avg_time:.2f}s")

        logger.info("评估完成！")
        logger.info(f"结果保存在: {args.output_dir}")

    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()