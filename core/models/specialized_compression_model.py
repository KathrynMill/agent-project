"""
专用剧本压缩模型 - 替换多智能体架构
基于深度学习的端到端剧本压缩系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """压缩级别枚举"""
    LIGHT = "light"      # 轻度压缩 70-90%
    MEDIUM = "medium"    # 中度压缩 50-70%
    HEAVY = "heavy"      # 重度压缩 30-50%
    CUSTOM = "custom"    # 自定义压缩


class PreservationType(Enum):
    """保护元素类型"""
    LOGIC = "logic"           # 逻辑关系
    STORY = "story"           # 故事线索
    CHARACTER = "character"   # 人物关系
    TIMELINE = "timeline"     # 时间线
    CLUE = "clue"            # 关键线索


@dataclass
class CompressionConfig:
    """压缩配置"""
    target_ratio: float = 0.6          # 目标压缩比例
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    preserve_elements: List[str] = None # 必须保留的元素
    logic_weight: float = 0.3          # 逻辑保持权重
    story_weight: float = 0.3          # 故事流畅权重
    playability_weight: float = 0.2    # 可玩性权重
    length_weight: float = 0.2         # 长度控制权重

    def __post_init__(self):
        if self.preserve_elements is None:
            self.preserve_elements = []


@dataclass
class ScriptElements:
    """剧本关键元素"""
    characters: List[str]      # 角色列表
    locations: List[str]       # 地点列表
    clues: List[str]          # 关键线索
    events: List[str]         # 重要事件
    timeline: List[str]       # 时间线事件
    relationships: List[str]  # 人物关系


@dataclass
class CompressionMetrics:
    """压缩质量指标"""
    compression_ratio: float     # 实际压缩比例
    logic_integrity: float       # 逻辑完整性得分 (0-1)
    story_coherence: float       # 故事连贯性得分 (0-1)
    playability_score: float     # 可玩性得分 (0-1)
    length_accuracy: float       # 长度控制准确性 (0-1)
    overall_quality: float       # 综合质量得分 (0-1)


class ScriptElementExtractor(nn.Module):
    """剧本元素提取器"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        # 元素分类头
        self.character_classifier = nn.Linear(hidden_size, 2)
        self.location_classifier = nn.Linear(hidden_size, 2)
        self.clue_classifier = nn.Linear(hidden_size, 2)
        self.event_classifier = nn.Linear(hidden_size, 2)
        self.relationship_classifier = nn.Linear(hidden_size, 2)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取剧本中的关键元素"""
        return {
            'characters': torch.sigmoid(self.character_classifier(hidden_states)),
            'locations': torch.sigmoid(self.location_classifier(hidden_states)),
            'clues': torch.sigmoid(self.clue_classifier(hidden_states)),
            'events': torch.sigmoid(self.event_classifier(hidden_states)),
            'relationships': torch.sigmoid(self.relationship_classifier(hidden_states))
        }


class LogicPreservationModule(nn.Module):
    """逻辑保持模块"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        # 逻辑一致性检查器
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # 因果关系建模
        self.causal_reasoning = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )

    def forward(self,
                original_hidden: torch.Tensor,
                compressed_hidden: torch.Tensor) -> torch.Tensor:
        """检查逻辑一致性"""
        # 原始和压缩表示的拼接
        combined = torch.cat([original_hidden, compressed_hidden], dim=-1)

        # 一致性得分
        consistency_score = self.consistency_checker(combined)

        # 因果关系推理
        causal_output, _ = self.causal_reasoning(
            compressed_hidden, compressed_hidden, compressed_hidden
        )

        return {
            'consistency_score': consistency_score,
            'causal_representation': causal_output
        }


class StoryCompressionModule(nn.Module):
    """故事压缩模块"""

    def __init__(self, hidden_size: int = 768, vocab_size: int = 32128):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # 基于T5的压缩生成器
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')

        # 压缩比例控制器
        self.ratio_controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 故事流畅性评估器
        self.fluency_evaluator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                target_ratio: float,
                encoder_hidden_states: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """执行故事压缩"""

        if encoder_hidden_states is None:
            # 使用T5编码器
            encoder_outputs = self.t5_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state

        # 调整压缩比例
        ratio_adjusted_hidden = encoder_hidden_states * (1 + target_ratio - 0.5)

        # 生成压缩后的文本
        decoder_input_ids = torch.zeros(
            (input_ids.size(0), 1),
            dtype=torch.long,
            device=input_ids.device
        )

        outputs = self.t5_model.generate(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            max_length=int(input_ids.size(1) * target_ratio),
            min_length=int(input_ids.size(1) * target_ratio * 0.8),
            num_beams=4,
            early_stopping=True,
            do_sample=True,
            temperature=0.8,
            top_p=0.92
        )

        # 计算流畅性得分
        fluency_score = self.fluency_evaluator(encoder_hidden_states.mean(dim=1))

        return {
            'compressed_ids': outputs,
            'encoder_hidden_states': encoder_hidden_states,
            'fluency_score': fluency_score,
            'ratio_adjusted_hidden': ratio_adjusted_hidden
        }


class QualityValidationModule(nn.Module):
    """质量验证模块"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        # 多任务质量评估器
        self.quality_heads = nn.ModuleDict({
            'logic_quality': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            'story_quality': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            'playability': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            'length_control': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
        })

        # 综合质量评估器
        self.overall_quality = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self,
                original_hidden: torch.Tensor,
                compressed_hidden: torch.Tensor,
                target_ratio: float,
                actual_ratio: float) -> Dict[str, torch.Tensor]:
        """评估压缩质量"""

        # 各项质量得分
        quality_scores = {}
        for name, head in self.quality_heads.items():
            combined = torch.cat([original_hidden, compressed_hidden], dim=-1).mean(dim=1)
            quality_scores[name] = head(combined)

        # 长度控制准确性
        length_accuracy = 1 - abs(target_ratio - actual_ratio) / target_ratio
        quality_scores['length_control'] = torch.tensor(
            [[length_accuracy]],
            device=original_hidden.device
        )

        # 综合质量得分
        quality_vector = torch.cat([
            quality_scores['logic_quality'],
            quality_scores['story_quality'],
            quality_scores['playability'],
            quality_scores['length_control']
        ], dim=-1)

        overall_score = self.overall_quality(quality_vector)

        return {
            **quality_scores,
            'overall_quality': overall_score
        }


class SpecializedCompressionModel(nn.Module):
    """
    专用剧本压缩模型
    整合元素提取、逻辑保持、故事压缩、质量验证于一体的端到端模型
    """

    def __init__(self,
                 model_name: str = 't5-large',
                 hidden_size: int = 768,
                 vocab_size: int = 32128):
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 核心模块
        self.element_extractor = ScriptElementExtractor(hidden_size)
        self.logic_preserver = LogicPreservationModule(hidden_size)
        self.story_compressor = StoryCompressionModule(hidden_size, vocab_size)
        self.quality_validator = QualityValidationModule(hidden_size)

        # 冻结部分T5参数以减少训练难度
        for param in self.story_compressor.t5_model.encoder.embeddings.parameters():
            param.requires_grad = False

    def forward(self,
                script_text: str,
                config: CompressionConfig) -> Dict[str, Any]:
        """前向传播 - 执行剧本压缩"""

        # 文本编码
        inputs = self.tokenizer(
            script_text,
            max_length=2048,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        # 获取T5编码器输出
        with torch.no_grad():
            encoder_outputs = self.story_compressor.t5_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state

        # 1. 元素提取
        elements = self.element_extractor(encoder_hidden_states)

        # 2. 故事压缩
        compression_result = self.story_compressor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ratio=config.target_ratio,
            encoder_hidden_states=encoder_hidden_states
        )

        # 3. 解码压缩结果
        compressed_text = self.tokenizer.decode(
            compression_result['compressed_ids'][0],
            skip_special_tokens=True
        )

        # 4. 重新编码压缩文本以计算质量指标
        compressed_inputs = self.tokenizer(
            compressed_text,
            max_length=2048,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        with torch.no_grad():
            compressed_encoder_outputs = self.story_compressor.t5_model.encoder(
                input_ids=compressed_inputs['input_ids'],
                attention_mask=compressed_inputs['attention_mask']
            )
            compressed_hidden_states = compressed_encoder_outputs.last_hidden_state

        # 5. 逻辑保持检查
        logic_result = self.logic_preserver(
            original_hidden=encoder_hidden_states,
            compressed_hidden=compressed_hidden_states
        )

        # 6. 质量验证
        actual_ratio = len(compressed_text) / len(script_text)
        quality_result = self.quality_validator(
            original_hidden=encoder_hidden_states,
            compressed_hidden=compressed_hidden_states,
            target_ratio=config.target_ratio,
            actual_ratio=actual_ratio
        )

        # 7. 构建结果
        metrics = CompressionMetrics(
            compression_ratio=actual_ratio,
            logic_integrity=logic_result['consistency_score'].item(),
            story_coherence=quality_result['story_quality'].item(),
            playability_score=quality_result['playability'].item(),
            length_accuracy=quality_result['length_control'].item(),
            overall_quality=quality_result['overall_quality'].item()
        )

        return {
            'compressed_text': compressed_text,
            'metrics': metrics,
            'elements': elements,
            'compression_result': compression_result,
            'logic_result': logic_result,
            'quality_result': quality_result
        }

    def extract_script_elements(self, script_text: str) -> ScriptElements:
        """提取剧本中的关键元素"""
        inputs = self.tokenizer(
            script_text,
            max_length=2048,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        with torch.no_grad():
            encoder_outputs = self.story_compressor.t5_model.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            hidden_states = encoder_outputs.last_hidden_state

            element_probabilities = self.element_extractor(hidden_states)

        # 转换概率为实际元素列表（简化版本）
        elements = ScriptElements(
            characters=self._extract_high_prob_elements(
                element_probabilities['characters'], script_text
            ),
            locations=self._extract_high_prob_elements(
                element_probabilities['locations'], script_text
            ),
            clues=self._extract_high_prob_elements(
                element_probabilities['clues'], script_text
            ),
            events=self._extract_high_prob_elements(
                element_probabilities['events'], script_text
            ),
            timeline=self._extract_high_prob_elements(
                element_probabilities['relationships'], script_text
            ),
            relationships=self._extract_high_prob_elements(
                element_probabilities['relationships'], script_text
            )
        )

        return elements

    def _extract_high_prob_elements(self,
                                   probabilities: torch.Tensor,
                                   text: str,
                                   threshold: float = 0.8) -> List[str]:
        """从概率分布中提取高置信度元素"""
        # 这是一个简化的实现，实际应用中需要更复杂的NLP技术
        tokens = self.tokenizer.tokenize(text)
        high_prob_indices = (probabilities.squeeze(-1) > threshold).nonzero().flatten()

        elements = []
        for idx in high_prob_indices:
            if idx < len(tokens):
                elements.append(tokens[idx])

        return list(set(elements))  # 去重

    def save_model(self, save_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size
        }, save_path)
        logger.info(f"模型已保存到: {save_path}")

    @classmethod
    def load_model(cls, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location='cpu')
        model = cls(
            model_name=checkpoint['model_name'],
            hidden_size=checkpoint['hidden_size'],
            vocab_size=checkpoint['vocab_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已从 {load_path} 加载")
        return model


def compute_compression_loss(config: CompressionConfig) -> nn.Module:
    """计算压缩模型的损失函数"""

    class CompressionLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,
                    model_outputs: Dict[str, Any],
                    target_text: str,
                    config: CompressionConfig) -> Dict[str, torch.Tensor]:

            # 1. 文本生成损失
            generation_loss = F.cross_entropy(
                model_outputs['compression_result']['compressed_ids'],
                target_text
            )

            # 2. 质量损失
            quality_loss = 0
            for metric_name in ['logic_integrity', 'story_coherence', 'playability_score']:
                target_quality = torch.tensor([1.0])  # 目标是完美质量
                current_quality = torch.tensor([
                    getattr(model_outputs['metrics'], metric_name)
                ])
                quality_loss += F.mse_loss(current_quality, target_quality)

            # 3. 长度控制损失
            length_loss = F.mse_loss(
                torch.tensor([model_outputs['metrics'].compression_ratio]),
                torch.tensor([config.target_ratio])
            )

            # 4. 总损失（加权组合）
            total_loss = (
                config.logic_weight * quality_loss +
                config.story_weight * generation_loss +
                config.playability_weight * quality_loss +
                config.length_weight * length_loss
            )

            return {
                'total_loss': total_loss,
                'generation_loss': generation_loss,
                'quality_loss': quality_loss,
                'length_loss': length_loss
            }

    return CompressionLoss()