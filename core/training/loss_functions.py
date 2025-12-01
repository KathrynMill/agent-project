"""
专用压缩模型的损失函数
包含多任务学习的复合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from ..models.specialized_compression_model import CompressionConfig, CompressionMetrics


class CompressionLoss(nn.Module):
    """
    专用压缩模型的复合损失函数
    结合文本生成损失、质量损失和长度控制损失
    """

    def __init__(self,
                 generation_weight: float = 0.4,
                 quality_weight: float = 0.3,
                 length_weight: float = 0.2,
                 regularity_weight: float = 0.1):
        """
        初始化损失函数

        Args:
            generation_weight: 文本生成损失权重
            quality_weight: 质量损失权重
            length_weight: 长度控制损失权重
            regularity_weight: 规律性损失权重
        """
        super().__init__()

        self.generation_weight = generation_weight
        self.quality_weight = quality_weight
        self.length_weight = length_weight
        self.regularity_weight = regularity_weight

        # 文本生成损失（交叉熵）
        self.generation_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # 质量评估损失（MSE）
        self.quality_criterion = nn.MSELoss()

        # 长度控制损失（L1损失）
        self.length_criterion = nn.L1Loss()

        # 一致性损失（余弦相似度）
        self.consistency_criterion = nn.CosineEmbeddingLoss()

        logger.info(f"复合损失函数初始化完成: gen={generation_weight}, qual={quality_weight}, "
                   f"len={length_weight}, reg={regularity_weight}")

    def forward(self,
                model_outputs: Dict[str, Any],
                target_config: CompressionConfig,
                reference_text: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        计算复合损失

        Args:
            model_outputs: 模型输出
            target_config: 目标压缩配置
            reference_text: 参考文本（可选）

        Returns:
            Dict[str, torch.Tensor]: 各项损失
        """
        device = next(iter(model_outputs.values())).device if isinstance(model_outputs, dict) else torch.device('cpu')

        # 提取模型输出
        compressed_text = model_outputs.get('compressed_text', '')
        metrics = model_outputs.get('metrics', CompressionMetrics(
            compression_ratio=0.5, logic_integrity=0.8, story_coherence=0.8,
            playability_score=0.8, length_accuracy=0.8, overall_quality=0.8
        ))

        # 1. 文本生成损失
        generation_loss = self._compute_generation_loss(model_outputs, reference_text)

        # 2. 质量损失
        quality_loss = self._compute_quality_loss(metrics)

        # 3. 长度控制损失
        length_loss = self._compute_length_loss(metrics, target_config)

        # 4. 规律性损失
        regularity_loss = self._compute_regularity_loss(compressed_text)

        # 5. 一致性损失（如果有参考文本）
        consistency_loss = torch.tensor(0.0, device=device)
        if reference_text and 'encoder_hidden_states' in model_outputs:
            consistency_loss = self._compute_consistency_loss(
                model_outputs['encoder_hidden_states'],
                model_outputs.get('ratio_adjusted_hidden', None)
            )

        # 加权总损失
        total_loss = (
            self.generation_weight * generation_loss +
            self.quality_weight * quality_loss +
            self.length_weight * length_loss +
            self.regularity_weight * regularity_loss +
            0.1 * consistency_loss  # 一致性损失权重固定为0.1
        )

        return {
            'total_loss': total_loss,
            'generation_loss': generation_loss,
            'quality_loss': quality_loss,
            'length_loss': length_loss,
            'regularity_loss': regularity_loss,
            'consistency_loss': consistency_loss
        }

    def _compute_generation_loss(self,
                                model_outputs: Dict[str, Any],
                                reference_text: Optional[str]) -> torch.Tensor:
        """计算文本生成损失"""
        if 'compressed_ids' not in model_outputs or reference_text is None:
            # 如果没有压缩ID或参考文本，返回零损失
            device = next(iter(model_outputs.values())).device if isinstance(model_outputs, dict) else torch.device('cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)

        compressed_ids = model_outputs['compressed_ids']
        device = compressed_ids.device

        # 这里简化处理，实际应用中需要完整的tokenization过程
        # 假设我们有一个tokenizer
        try:
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained('t5-large')

            # 编码参考文本
            target_ids = tokenizer.encode(
                reference_text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding='max_length'
            ).to(device)

            # 计算交叉熵损失
            # compressed_ids: [batch_size, seq_len]
            # target_ids: [batch_size, seq_len]
            loss = self.generation_criterion(
                compressed_ids.view(-1, compressed_ids.size(-1)),
                target_ids.view(-1)
            )

            return loss

        except Exception as e:
            logger.warning(f"文本生成损失计算失败: {e}")
            return torch.tensor(0.0, device=device, requires_grad=True)

    def _compute_quality_loss(self, metrics: CompressionMetrics) -> torch.Tensor:
        """计算质量损失"""
        # 目标质量分数（完美质量）
        target_quality = torch.tensor([
            metrics.logic_integrity,      # 逻辑完整性应该保持
            metrics.story_coherence,      # 故事连贯性应该保持
            metrics.playability_score,    # 可玩性应该保持
            metrics.overall_quality       # 整体质量应该保持
        ], dtype=torch.float32)

        # 当前质量分数
        current_quality = torch.tensor([
            metrics.logic_integrity,
            metrics.story_coherence,
            metrics.playability_score,
            metrics.overall_quality
        ], dtype=torch.float32)

        # 计算质量损失（希望接近目标质量，但不要过拟合）
        quality_loss = self.quality_criterion(current_quality, target_quality)

        # 额外的质量惩罚：如果质量低于阈值，增加惩罚
        quality_threshold = 0.7
        if metrics.overall_quality < quality_threshold:
            quality_penalty = (quality_threshold - metrics.overall_quality) * 2.0
            quality_loss = quality_loss + torch.tensor(quality_penalty, dtype=torch.float32)

        return quality_loss

    def _compute_length_loss(self,
                           metrics: CompressionMetrics,
                           target_config: CompressionConfig) -> torch.Tensor:
        """计算长度控制损失"""
        target_ratio = torch.tensor([target_config.target_ratio], dtype=torch.float32)
        current_ratio = torch.tensor([metrics.compression_ratio], dtype=torch.float32)

        # L1损失：控制压缩比例
        length_loss = self.length_criterion(current_ratio, target_ratio)

        # 额外的长度控制惩罚：如果偏离目标比例太多，增加惩罚
        deviation = abs(metrics.compression_ratio - target_config.target_ratio)
        max_allowed_deviation = 0.1

        if deviation > max_allowed_deviation:
            length_penalty = (deviation - max_allowed_deviation) * 5.0
            length_loss = length_loss + torch.tensor(length_penalty, dtype=torch.float32)

        return length_loss

    def _compute_regularity_loss(self, compressed_text: str) -> torch.Tensor:
        """计算规律性损失"""
        if not compressed_text:
            return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        # 1. 句子长度一致性损失
        sentences = [s.strip() for s in compressed_text.split('。') if s.strip()]
        if len(sentences) > 1:
            sentence_lengths = [len(s) for s in sentences]
            mean_length = sum(sentence_lengths) / len(sentence_lengths)
            length_variance = sum((l - mean_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            sentence_regularization = torch.tensor(length_variance / 1000, dtype=torch.float32)  # 归一化
        else:
            sentence_regularization = torch.tensor(0.0, dtype=torch.float32)

        # 2. 词汇重复性损失
        words = compressed_text.split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_penalty = torch.tensor(1.0 - unique_words / len(words), dtype=torch.float32)
        else:
            repetition_penalty = torch.tensor(0.0, dtype=torch.float32)

        # 3. 标点符号一致性损失
        punctuation_ratio = compressed_text.count('。') / max(1, len(compressed_text))
        target_punctuation_ratio = 0.02  # 大约每50个字符一个句号
        punctuation_loss = torch.tensor(abs(punctuation_ratio - target_punctuation_ratio), dtype=torch.float32)

        # 总规律性损失
        regularity_loss = (
            0.5 * sentence_regularization +
            0.3 * repetition_penalty +
            0.2 * punctuation_loss
        )

        return regularity_loss

    def _compute_consistency_loss(self,
                                original_hidden: torch.Tensor,
                                compressed_hidden: Optional[torch.Tensor]) -> torch.Tensor:
        """计算一致性损失"""
        if compressed_hidden is None:
            return torch.tensor(0.0, dtype=torch.float32, device=original_hidden.device)

        # 计算原始文本和压缩文本表示的余弦相似度
        # original_hidden: [batch_size, seq_len, hidden_size]
        # compressed_hidden: [batch_size, seq_len, hidden_size]

        # 池化：取序列的平均值
        original_pooled = original_hidden.mean(dim=1)  # [batch_size, hidden_size]
        compressed_pooled = compressed_hidden.mean(dim=1)  # [batch_size, hidden_size]

        # 计算相似度目标（希望保持一定相似性）
        target_similarity = torch.ones(original_pooled.size(0), dtype=torch.float32, device=original_hidden.device)

        # 计算余弦一致性损失
        consistency_loss = self.consistency_criterion(
            original_pooled,
            compressed_pooled,
            target_similarity
        )

        return consistency_loss


class AdaptiveCompressionLoss(nn.Module):
    """自适应压缩损失函数"""

    def __init__(self,
                 initial_weights: Dict[str, float] = None,
                 adaptation_rate: float = 0.01):
        """
        初始化自适应损失

        Args:
            initial_weights: 初始权重
            adaptation_rate: 自适应学习率
        """
        super().__init__()

        self.initial_weights = initial_weights or {
            'generation': 0.4,
            'quality': 0.3,
            'length': 0.2,
            'regularity': 0.1
        }
        self.adaptation_rate = adaptation_rate

        # 可学习的权重参数
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in self.initial_weights.items()
        })

        # 权重历史（用于平滑）
        self.weight_history = {name: [] for name in self.initial_weights.keys()}

        # 基础损失函数
        self.base_loss = CompressionLoss()

    def forward(self,
                model_outputs: Dict[str, Any],
                target_config: CompressionConfig,
                current_epoch: int = 0,
                reference_text: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        计算自适应损失

        Args:
            model_outputs: 模型输出
            target_config: 目标配置
            current_epoch: 当前训练轮数
            reference_text: 参考文本

        Returns:
            Dict[str, torch.Tensor]: 损失结果
        """
        # 计算基础损失
        base_losses = self.base_loss(model_outputs, target_config, reference_text)

        # 自适应调整权重
        adapted_weights = self._adapt_weights(base_losses, current_epoch)

        # 计算加权总损失
        total_loss = (
            adapted_weights['generation'] * base_losses['generation_loss'] +
            adapted_weights['quality'] * base_losses['quality_loss'] +
            adapted_weights['length'] * base_losses['length_loss'] +
            adapted_weights['regularity'] * base_losses['regularity_loss'] +
            0.1 * base_losses['consistency_loss']
        )

        # 添加权重正则化损失（防止权重过大）
        weight_regularization = sum(
            torch.abs(weight - initial_weight)
            for (name, weight), initial_weight in zip(self.weights.items(), self.initial_weights.values())
        ) * 0.01

        total_loss = total_loss + weight_regularization

        # 记录权重历史
        for name in self.weights.keys():
            self.weight_history[name].append(adapted_weights[name].item())

        # 返回结果
        result = base_losses.copy()
        result['total_loss'] = total_loss
        result['adapted_weights'] = adapted_weights
        result['weight_regularization'] = weight_regularization

        return result

    def _adapt_weights(self,
                      losses: Dict[str, torch.Tensor],
                      current_epoch: int) -> Dict[str, float]:
        """自适应调整权重"""
        adapted_weights = {}

        # 基于损失大小调整权重
        for name in self.weights.keys():
            base_weight = self.initial_weights[name]
            current_weight = self.weights[name].item()

            # 计算损失梯度（简化版本）
            loss_name = f"{name}_loss"
            if loss_name in losses:
                loss_value = losses[loss_name].item()

                # 如果损失很大，增加权重；如果损失很小，减少权重
                if loss_value > 1.0:
                    adjustment = self.adaptation_rate
                elif loss_value < 0.1:
                    adjustment = -self.adaptation_rate
                else:
                    adjustment = 0.0

                # 应用调整
                new_weight = current_weight + adjustment
                new_weight = max(0.01, min(1.0, new_weight))  # 限制在[0.01, 1.0]范围内

                adapted_weights[name] = new_weight
            else:
                adapted_weights[name] = base_weight

        # 归一化权重
        total_weight = sum(adapted_weights.values())
        if total_weight > 1.0:
            for name in adapted_weights:
                adapted_weights[name] /= total_weight

        return adapted_weights

    def get_weight_statistics(self) -> Dict[str, Any]:
        """获取权重统计信息"""
        stats = {
            'current_weights': {name: weight.item() for name, weight in self.weights.items()},
            'initial_weights': self.initial_weights,
            'weight_changes': {}
        }

        # 计算权重变化
        for name in self.weights.keys():
            if len(self.weight_history[name]) > 0:
                initial = self.initial_weights[name]
                current = self.weights[name].item()
                change = current - initial
                stats['weight_changes'][name] = {
                    'absolute_change': change,
                    'relative_change': change / initial if initial != 0 else 0.0,
                    'history_length': len(self.weight_history[name])
                }

        return stats


class LossScheduler:
    """损失调度器"""

    def __init__(self,
                 schedule_type: str = "linear",
                 total_epochs: int = 50,
                 warmup_epochs: int = 5):
        """
        初始化损失调度器

        Args:
            schedule_type: 调度类型 ('linear', 'cosine', 'exponential')
            total_epochs: 总训练轮数
            warmup_epochs: 预热轮数
        """
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_weight_multipliers(self, current_epoch: int) -> Dict[str, float]:
        """获取当前轮数的权重乘数"""
        if current_epoch < self.warmup_epochs:
            # 预热阶段：逐步增加生成损失权重
            progress = current_epoch / self.warmup_epochs
            return {
                'generation': 0.1 + 0.9 * progress,
                'quality': 0.5,
                'length': 0.3,
                'regularity': 0.2
            }

        # 训练阶段：根据调度类型调整
        progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

        if self.schedule_type == "linear":
            # 线性调度：逐渐减少生成损失权重，增加质量权重
            return {
                'generation': 0.4 * (1 - progress * 0.5),
                'quality': 0.3 * (1 + progress * 0.5),
                'length': 0.2,
                'regularity': 0.2
            }
        elif self.schedule_type == "cosine":
            # 余弦调度
            generation_weight = 0.4 * (0.5 + 0.5 * math.cos(math.pi * progress))
            quality_weight = 0.3 * (1.5 - 0.5 * math.cos(math.pi * progress))
            return {
                'generation': generation_weight,
                'quality': quality_weight,
                'length': 0.2,
                'regularity': 0.2
            }
        elif self.schedule_type == "exponential":
            # 指数调度：快速转向质量优化
            generation_weight = 0.4 * math.exp(-2 * progress)
            quality_weight = 0.3 * (1 + 2 * (1 - math.exp(-2 * progress)))
            return {
                'generation': generation_weight,
                'quality': quality_weight,
                'length': 0.2,
                'regularity': 0.2
            }
        else:
            # 默认：固定权重
            return {
                'generation': 0.4,
                'quality': 0.3,
                'length': 0.2,
                'regularity': 0.2
            }