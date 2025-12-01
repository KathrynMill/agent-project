"""
数据处理管道 - 专用模型训练数据准备
剧本压缩模型的训练数据生成、预处理和批处理
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import random
import re
from pathlib import Path
import logging
from transformers import T5Tokenizer

from ..models.specialized_compression_model import (
    CompressionConfig, CompressionLevel, ScriptElements, CompressionMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """训练样本数据结构"""
    script_id: str
    original_script: str
    compressed_script: str
    compression_level: CompressionLevel
    compression_ratio: float
    preserved_elements: List[str]
    logic_integrity: float
    story_coherence: float
    playability_score: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidationResult:
    """数据验证结果"""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ScriptDataset(Dataset):
    """剧本数据集"""

    def __init__(self,
                 examples: List[TrainingExample],
                 tokenizer: T5Tokenizer,
                 max_length: int = 2048,
                 include_metrics: bool = True):
        """
        初始化数据集

        Args:
            examples: 训练样本列表
            tokenizer: 文本tokenizer
            max_length: 最大序列长度
            include_metrics: 是否包含质量指标
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_metrics = include_metrics

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # 编码原始剧本
        original_inputs = self.tokenizer(
            example.original_script,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # 编码压缩剧本
        compressed_inputs = self.tokenizer(
            example.compressed_script,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        result = {
            'script_id': example.script_id,
            'original_input_ids': original_inputs['input_ids'].squeeze(),
            'original_attention_mask': original_inputs['attention_mask'].squeeze(),
            'compressed_input_ids': compressed_inputs['input_ids'].squeeze(),
            'compressed_attention_mask': compressed_inputs['attention_mask'].squeeze(),
            'target_ratio': torch.tensor([example.compression_ratio], dtype=torch.float),
            'compression_level': torch.tensor([
                list(CompressionLevel).index(example.compression_level)
            ], dtype=torch.long)
        }

        if self.include_metrics:
            result.update({
                'logic_integrity': torch.tensor([example.logic_integrity], dtype=torch.float),
                'story_coherence': torch.tensor([example.story_coherence], dtype=torch.float),
                'playability_score': torch.tensor([example.playability_score], dtype=torch.float)
            })

        return result


class DataCollector:
    """训练数据收集器"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_json(self, json_file: str) -> List[TrainingExample]:
        """从JSON文件收集训练数据"""
        examples = []

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                example = TrainingExample(
                    script_id=item['script_id'],
                    original_script=item['original_script'],
                    compressed_script=item['compressed_script'],
                    compression_level=CompressionLevel(item['compression_level']),
                    compression_ratio=item['compression_ratio'],
                    preserved_elements=item.get('preserved_elements', []),
                    logic_integrity=item.get('logic_integrity', 0.8),
                    story_coherence=item.get('story_coherence', 0.8),
                    playability_score=item.get('playability_score', 0.8),
                    metadata=item.get('metadata', {})
                )
                examples.append(example)

            logger.info(f"从 {json_file} 收集到 {len(examples)} 个训练样本")

        except Exception as e:
            logger.error(f"从JSON文件收集数据失败: {e}")

        return examples

    def collect_from_directory(self, directory: str) -> List[TrainingExample]:
        """从目录收集多个JSON文件的数据"""
        all_examples = []
        directory_path = Path(directory)

        for json_file in directory_path.glob('*.json'):
            examples = self.collect_from_json(str(json_file))
            all_examples.extend(examples)

        logger.info(f"从目录 {directory} 共收集到 {len(all_examples)} 个训练样本")
        return all_examples

    def save_examples(self, examples: List[TrainingExample], output_file: str):
        """保存训练样本到文件"""
        data = [asdict(example) for example in examples]

        # 转换枚举为字符串
        for item in data:
            item['compression_level'] = item['compression_level'].value

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"已保存 {len(examples)} 个训练样本到 {output_file}")

    def load_examples(self, input_file: str) -> List[TrainingExample]:
        """从文件加载训练样本"""
        examples = []

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                example = TrainingExample(
                    script_id=item['script_id'],
                    original_script=item['original_script'],
                    compressed_script=item['compressed_script'],
                    compression_level=CompressionLevel(item['compression_level']),
                    compression_ratio=item['compression_ratio'],
                    preserved_elements=item.get('preserved_elements', []),
                    logic_integrity=item.get('logic_integrity', 0.8),
                    story_coherence=item.get('story_coherence', 0.8),
                    playability_score=item.get('playability_score', 0.8),
                    metadata=item.get('metadata', {})
                )
                examples.append(example)

            logger.info(f"从 {input_file} 加载了 {len(examples)} 个训练样本")

        except Exception as e:
            logger.error(f"加载训练样本失败: {e}")

        return examples


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_example(example: TrainingExample) -> ValidationResult:
        """验证单个训练样本"""
        warnings = []

        # 检查基本字段
        if not example.script_id:
            return ValidationResult(False, "script_id 不能为空", warnings)

        if not example.original_script.strip():
            return ValidationResult(False, "原始剧本不能为空", warnings)

        if not example.compressed_script.strip():
            return ValidationResult(False, "压缩剧本不能为空", warnings)

        # 检查压缩比例
        actual_ratio = len(example.compressed_script) / len(example.original_script)
        ratio_diff = abs(actual_ratio - example.compression_ratio)

        if ratio_diff > 0.1:  # 允许10%的误差
            warnings.append(f"压缩比例不准确: 实际={actual_ratio:.3f}, 声明={example.compression_ratio:.3f}")

        # 检查压缩比例范围
        if not (0.3 <= actual_ratio <= 0.9):
            warnings.append(f"压缩比例超出合理范围: {actual_ratio:.3f}")

        # 检查质量指标
        for score_name, score_value in [
            ('logic_integrity', example.logic_integrity),
            ('story_coherence', example.story_coherence),
            ('playability_score', example.playability_score)
        ]:
            if not (0.0 <= score_value <= 1.0):
                warnings.append(f"{score_name} 超出范围 [0,1]: {score_value}")

        # 检查剧本长度
        if len(example.original_script) < 100:
            warnings.append("原始剧本过短 (<100字符)")

        if len(example.compressed_script) < 50:
            warnings.append("压缩剧本过短 (<50字符)")

        return ValidationResult(True, None, warnings)

    @staticmethod
    def validate_dataset(examples: List[TrainingExample]) -> Dict[str, Any]:
        """验证整个数据集"""
        validation_results = {
            'total_examples': len(examples),
            'valid_examples': 0,
            'invalid_examples': 0,
            'warnings': [],
            'statistics': {
                'compression_ratios': [],
                'quality_scores': {
                    'logic_integrity': [],
                    'story_coherence': [],
                    'playability_score': []
                },
                'compression_levels': {},
                'script_lengths': []
            }
        }

        script_ids = set()

        for example in examples:
            # 检查重复ID
            if example.script_id in script_ids:
                validation_results['warnings'].append(f"重复的script_id: {example.script_id}")
            script_ids.add(example.script_id)

            # 验证单个样本
            result = DataValidator.validate_example(example)

            if result.is_valid:
                validation_results['valid_examples'] += 1
            else:
                validation_results['invalid_examples'] += 1
                validation_results['warnings'].append(f"无效样本 {example.script_id}: {result.error_message}")

            validation_results['warnings'].extend([
                f"{example.script_id}: {warning}" for warning in result.warnings
            ])

            # 收集统计信息
            validation_results['statistics']['compression_ratios'].append(example.compression_ratio)
            validation_results['statistics']['quality_scores']['logic_integrity'].append(example.logic_integrity)
            validation_results['statistics']['quality_scores']['story_coherence'].append(example.story_coherence)
            validation_results['statistics']['quality_scores']['playability_score'].append(example.playability_score)

            level_name = example.compression_level.value
            validation_results['statistics']['compression_levels'][level_name] = \
                validation_results['statistics']['compression_levels'].get(level_name, 0) + 1

            validation_results['statistics']['script_lengths'].append(len(example.original_script))

        return validation_results


class DataAugmenter:
    """数据增强器"""

    def __init__(self):
        self.augmentation_methods = [
            self._paraphrase_original,
            self._vary_compression_ratio,
            self._shuffle_preserved_elements,
            self._add_noise_to_quality_scores
        ]

    def augment_example(self, example: TrainingExample, num_augmented: int = 3) -> List[TrainingExample]:
        """增强单个训练样本"""
        augmented_examples = []

        for _ in range(num_augmented):
            # 随机选择增强方法
            method = random.choice(self.augmentation_methods)
            augmented = method(example.copy()) if hasattr(example, 'copy') else method(example)
            augmented_examples.append(augmented)

        return augmented_examples

    def augment_dataset(self, examples: List[TrainingExample], augmentation_factor: float = 2.0) -> List[TrainingExample]:
        """增强整个数据集"""
        augmented_examples = examples.copy()
        num_to_augment = int(len(examples) * (augmentation_factor - 1.0))

        selected_examples = random.choices(examples, k=num_to_augment)

        for example in selected_examples:
            augmented_list = self.augment_example(example, num_augmented=1)
            augmented_examples.extend(augmented_list)

        logger.info(f"数据增强完成: {len(examples)} -> {len(augmented_examples)} 个样本")
        return augmented_examples

    def _paraphrase_original(self, example: TrainingExample) -> TrainingExample:
        """转述原始剧本（简化版本）"""
        # 这里可以使用更复杂的NLP技术
        text = example.original_script
        # 简单的同义词替换（实际应用中应该使用更高级的方法）
        paraphrased = text.replace('然后', '接着').replace('因为', '由于')

        example.original_script = paraphrased
        example.script_id = f"{example.script_id}_paraphrased"
        return example

    def _vary_compression_ratio(self, example: TrainingExample) -> TrainingExample:
        """变化压缩比例"""
        variation = random.uniform(-0.1, 0.1)
        new_ratio = max(0.3, min(0.9, example.compression_ratio + variation))

        example.compression_ratio = new_ratio
        example.script_id = f"{example.script_id}_ratio_{new_ratio:.2f}"
        return example

    def _shuffle_preserved_elements(self, example: TrainingExample) -> TrainingExample:
        """随机化保留元素"""
        if example.preserved_elements:
            random.shuffle(example.preserved_elements)

        example.script_id = f"{example.script_id}_shuffled"
        return example

    def _add_noise_to_quality_scores(self, example: TrainingExample) -> TrainingExample:
        """为质量分数添加噪声"""
        noise_factor = 0.05

        example.logic_integrity = max(0.0, min(1.0, example.logic_integrity + random.uniform(-noise_factor, noise_factor)))
        example.story_coherence = max(0.0, min(1.0, example.story_coherence + random.uniform(-noise_factor, noise_factor)))
        example.playability_score = max(0.0, min(1.0, example.playability_score + random.uniform(-noise_factor, noise_factor)))

        example.script_id = f"{example.script_id}_noisy"
        return example


class DataSplitter:
    """数据集分割器"""

    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.15, test_ratio: float = 0.05):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_by_ratio(self, examples: List[TrainingExample], shuffle: bool = True) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """按比例分割数据集"""
        if shuffle:
            examples = examples.copy()
            random.shuffle(examples)

        total = len(examples)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)

        train_examples = examples[:train_size]
        val_examples = examples[train_size:train_size + val_size]
        test_examples = examples[train_size + val_size:]

        logger.info(f"数据集分割完成: 训练={len(train_examples)}, 验证={len(val_examples)}, 测试={len(test_examples)}")
        return train_examples, val_examples, test_examples

    def split_by_compression_level(self, examples: List[TrainingExample]) -> Dict[str, List[TrainingExample]]:
        """按压缩级别分组"""
        groups = {}

        for example in examples:
            level = example.compression_level.value
            if level not in groups:
                groups[level] = []
            groups[level].append(example)

        logger.info(f"按压缩级别分组: { {k: len(v) for k, v in groups.items()} }")
        return groups


class DataPipeline:
    """数据处理管道主类"""

    def __init__(self,
                 data_dir: str,
                 tokenizer_name: str = 't5-large',
                 max_length: int = 2048,
                 batch_size: int = 8,
                 num_workers: int = 4):
        """
        初始化数据管道

        Args:
            data_dir: 数据目录
            tokenizer_name: tokenizer名称
            max_length: 最大序列长度
            batch_size: 批大小
            num_workers: 数据加载进程数
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 组件
        self.collector = DataCollector(str(self.data_dir / "raw"))
        self.validator = DataValidator()
        self.augmenter = DataAugmenter()
        self.splitter = DataSplitter()

    def prepare_training_data(self,
                             raw_data_sources: List[str],
                             output_dir: str,
                             apply_augmentation: bool = True,
                             augmentation_factor: float = 2.0) -> Dict[str, str]:
        """
        准备训练数据

        Args:
            raw_data_sources: 原始数据源列表（JSON文件或目录）
            output_dir: 输出目录
            apply_augmentation: 是否应用数据增强
            augmentation_factor: 增强倍数

        Returns:
            Dict[str, str]: 分割后的数据文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. 收集原始数据
        all_examples = []
        for source in raw_data_sources:
            if os.path.isfile(source):
                examples = self.collector.collect_from_json(source)
            else:
                examples = self.collector.collect_from_directory(source)
            all_examples.extend(examples)

        if not all_examples:
            raise ValueError("没有找到任何训练数据")

        logger.info(f"总共收集到 {len(all_examples)} 个原始训练样本")

        # 2. 验证数据
        validation_result = self.validator.validate_dataset(all_examples)

        if validation_result['invalid_examples'] > 0:
            logger.warning(f"发现 {validation_result['invalid_examples']} 个无效样本")

        for warning in validation_result['warnings'][:10]:  # 只显示前10个警告
            logger.warning(warning)

        # 3. 数据增强
        if apply_augmentation:
            all_examples = self.augmenter.augment_dataset(
                all_examples, augmentation_factor
            )

        # 4. 分割数据集
        train_examples, val_examples, test_examples = self.splitter.split_by_ratio(all_examples)

        # 5. 保存处理后的数据
        train_file = output_path / "train.json"
        val_file = output_path / "val.json"
        test_file = output_path / "test.json"

        self.collector.save_examples(train_examples, str(train_file))
        self.collector.save_examples(val_examples, str(val_file))
        self.collector.save_examples(test_examples, str(test_file))

        logger.info("训练数据准备完成")

        return {
            'train': str(train_file),
            'val': str(val_file),
            'test': str(test_file)
        }

    def create_data_loaders(self,
                           data_files: Dict[str, str]) -> Dict[str, DataLoader]:
        """创建数据加载器"""
        loaders = {}

        for split, file_path in data_files.items():
            # 加载训练样本
            examples = self.collector.load_examples(file_path)
            if not examples:
                logger.warning(f"数据文件 {file_path} 为空，跳过")
                continue

            # 创建数据集
            dataset = ScriptDataset(
                examples=examples,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                include_metrics=(split != 'test')  # 测试集不包含质量指标
            )

            # 创建数据加载器
            shuffle = (split == 'train')
            drop_last = (split == 'train')

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=drop_last,
                pin_memory=torch.cuda.is_available()
            )

            loaders[split] = loader
            logger.info(f"创建了 {split} 数据加载器: {len(dataset)} 个样本")

        return loaders

    def get_dataset_statistics(self, data_files: Dict[str, str]) -> Dict[str, Any]:
        """获取数据集统计信息"""
        statistics = {}

        for split, file_path in data_files.items():
            examples = self.collector.load_examples(file_path)
            validation_result = self.validator.validate_dataset(examples)

            stats = validation_result['statistics']
            stats.update({
                'total_examples': len(examples),
                'valid_examples': validation_result['valid_examples'],
                'invalid_examples': validation_result['invalid_examples']
            })

            # 计算平均值
            if stats['compression_ratios']:
                stats['avg_compression_ratio'] = sum(stats['compression_ratios']) / len(stats['compression_ratios'])

            if stats['script_lengths']:
                stats['avg_script_length'] = sum(stats['script_lengths']) / len(stats['script_lengths'])

            statistics[split] = stats

        return statistics

    def create_sample_data(self, output_file: str, num_samples: int = 100):
        """创建示例训练数据（用于测试）"""
        sample_examples = []

        for i in range(num_samples):
            # 生成模拟剧本
            script_length = random.randint(500, 2000)
            original_script = f"这是第{i+1}个剧本杀的原始内容。" + "A说了一句话。B回应了。" * (script_length // 20)

            # 生成压缩版本
            compression_ratio = random.uniform(0.4, 0.8)
            compressed_length = int(script_length * compression_ratio)
            compressed_script = f"压缩后的剧本{i+1}。" + "A对话。B回答。" * (compressed_length // 10)

            example = TrainingExample(
                script_id=f"sample_{i+1:04d}",
                original_script=original_script,
                compressed_script=compressed_script,
                compression_level=random.choice(list(CompressionLevel)),
                compression_ratio=compression_ratio,
                preserved_elements=[f"关键线索{j}" for j in range(random.randint(1, 5))],
                logic_integrity=random.uniform(0.7, 0.95),
                story_coherence=random.uniform(0.7, 0.95),
                playability_score=random.uniform(0.7, 0.95),
                metadata={'is_sample': True, 'generated': True}
            )
            sample_examples.append(example)

        self.collector.save_examples(sample_examples, output_file)
        logger.info(f"已创建 {num_samples} 个示例训练样本: {output_file}")


# 使用示例
if __name__ == "__main__":
    # 创建数据管道
    pipeline = DataPipeline(data_dir="./data")

    # 创建示例数据
    pipeline.create_sample_data("./data/samples/sample_data.json", num_samples=50)

    # 准备训练数据
    data_files = pipeline.prepare_training_data(
        raw_data_sources=["./data/samples/sample_data.json"],
        output_dir="./data/processed",
        apply_augmentation=True,
        augmentation_factor=1.5
    )

    # 创建数据加载器
    loaders = pipeline.create_data_loaders(data_files)

    # 获取统计信息
    stats = pipeline.get_dataset_statistics(data_files)
    print("数据集统计信息:", json.dumps(stats, indent=2, ensure_ascii=False))