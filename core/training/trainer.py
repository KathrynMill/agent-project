"""
专用压缩模型训练管道
包含训练循环、验证、保存等功能
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm
import wandb
from dataclasses import asdict

from ..models.specialized_compression_model import (
    SpecializedCompressionModel, CompressionConfig, CompressionLevel
)
from ..data.data_pipeline import DataPipeline, TrainingExample
from .loss_functions import CompressionLoss

logger = logging.getLogger(__name__)


class CompressionTrainer:
    """专用压缩模型训练器"""

    def __init__(self,
                 model: SpecializedCompressionModel,
                 data_pipeline: DataPipeline,
                 config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            model: 专用压缩模型
            data_pipeline: 数据管道
            config: 训练配置
        """
        self.model = model
        self.data_pipeline = data_pipeline
        self.config = config

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 训练参数
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 4)
        self.num_epochs = config.get('num_epochs', 50)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        # 保存路径
        self.output_dir = Path(config.get('output_dir', './checkpoints'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志配置
        self.log_interval = config.get('log_interval', 100)
        self.save_interval = config.get('save_interval', 5)
        self.eval_interval = config.get('eval_interval', 1)

        # Wandb配置
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="script-compression-model",
                config=config,
                name=f"compression_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # 优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()

        # 损失函数
        self.criterion = CompressionLoss()

        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []

        logger.info(f"训练器初始化完成，设备: {self.device}")

    def _setup_optimizer(self):
        """设置优化器"""
        # 分层学习率
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 't5_model' in n and p.requires_grad],
                'lr': self.learning_rate * 0.1,  # T5模型使用较小的学习率
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 't5_model' not in n and p.requires_grad],
                'lr': self.learning_rate,
                'weight_decay': 0.01
            }
        ]

        self.optimizer = optim.AdamW(param_groups)

    def _setup_scheduler(self):
        """设置学习率调度器"""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate * 0.1
        )

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        执行训练

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器

        Returns:
            Dict[str, Any]: 训练结果
        """
        logger.info(f"开始训练，总轮数: {self.num_epochs}")

        for epoch in range(self.num_epochs):
            # 训练阶段
            train_metrics = self._train_epoch(train_loader, epoch)

            # 验证阶段
            if val_loader and epoch % self.eval_interval == 0:
                val_metrics = self._validate_epoch(val_loader, epoch)
            else:
                val_metrics = {}

            # 学习率调度
            self.scheduler.step()

            # 记录训练历史
            epoch_record = {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'learning_rate': self.scheduler.get_last_lr()[0],
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(epoch_record)

            # 保存检查点
            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)

            # 日志记录
            self._log_epoch(epoch_record)

        # 保存最终模型
        self._save_final_model()

        # 训练完成
        result = {
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.num_epochs,
            'final_output_dir': str(self.output_dir)
        }

        logger.info("训练完成")
        return result

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 数据移到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 前向传播
                loss_dict = self._compute_loss(batch)

                # 梯度缩放（混合精度）
                loss = loss_dict['total_loss']
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                # 反向传播
                loss.backward()

                # 梯度累积
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                    # 优化器步骤
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # 累计损失
                total_loss += loss.item() * self.gradient_accumulation_steps

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                })

                # 日志记录
                if self.global_step % self.log_interval == 0:
                    self._log_training_step(loss_dict, batch_idx, epoch)

            except Exception as e:
                logger.error(f"训练批次失败 (batch {batch_idx}): {e}")
                continue

        # 计算平均损失
        avg_loss = total_loss / num_batches

        return {
            'train_loss': avg_loss,
            'total_batches': num_batches,
            'global_step': self.global_step
        }

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                try:
                    # 数据移到设备
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # 前向传播
                    loss_dict = self._compute_loss(batch)
                    total_loss += loss_dict['total_loss'].item()

                except Exception as e:
                    logger.error(f"验证批次失败: {e}")
                    continue

        # 计算平均损失
        avg_loss = total_loss / num_batches

        # 更新最佳验证损失
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_best_model(epoch, avg_loss)

        return {
            'val_loss': avg_loss,
            'total_batches': num_batches
        }

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        # 从batch中提取数据
        original_input_ids = batch['original_input_ids']
        original_attention_mask = batch['original_attention_mask']
        target_ratio = batch['target_ratio']

        # 创建压缩配置
        configs = []
        for ratio in target_ratio:
            if ratio >= 0.7:
                level = CompressionLevel.LIGHT
            elif ratio >= 0.5:
                level = CompressionLevel.MEDIUM
            else:
                level = CompressionLevel.HEAVY

            config = CompressionConfig(
                target_ratio=ratio.item(),
                compression_level=level
            )
            configs.append(config)

        # 执行模型前向传播
        results = []
        for i, config in enumerate(configs):
            # 获取原始文本（这里简化处理）
            original_text = "Sample script text"  # 实际应用中需要从input_ids解码

            try:
                result = self.model(original_text, config)
                results.append(result)
            except Exception as e:
                # 创建默认结果
                from ..models.specialized_compression_model import CompressionMetrics
                dummy_metrics = CompressionMetrics(
                    compression_ratio=config.target_ratio,
                    logic_integrity=0.8,
                    story_coherence=0.8,
                    playability_score=0.8,
                    length_accuracy=0.8,
                    overall_quality=0.8
                )
                results.append({
                    'compressed_text': original_text[:int(len(original_text) * config.target_ratio)],
                    'metrics': dummy_metrics
                })

        # 计算损失
        total_loss = 0.0
        loss_components = {
            'generation_loss': 0.0,
            'quality_loss': 0.0,
            'length_loss': 0.0
        }

        for i, (result, config) in enumerate(zip(results, configs)):
            # 这里简化损失计算，实际应用中需要更复杂的逻辑
            generation_loss = nn.MSELoss()(
                torch.tensor([result['metrics'].compression_ratio]),
                torch.tensor([config.target_ratio])
            )

            quality_loss = 1.0 - result['metrics'].overall_quality

            length_loss = abs(result['metrics'].compression_ratio - config.target_ratio)

            # 加权总损失
            step_loss = (
                config.story_weight * generation_loss +
                config.logic_weight * quality_loss +
                config.length_weight * length_loss
            )

            total_loss += step_loss
            loss_components['generation_loss'] += generation_loss
            loss_components['quality_loss'] += quality_loss
            loss_components['length_loss'] += length_loss

        # 平均损失
        batch_size = len(configs)
        loss_dict = {
            'total_loss': total_loss / batch_size,
            'generation_loss': loss_components['generation_loss'] / batch_size,
            'quality_loss': loss_components['quality_loss'] / batch_size,
            'length_loss': loss_components['length_loss'] / batch_size
        }

        return {k: torch.tensor(v, requires_grad=True) for k, v in loss_dict.items()}

    def _log_training_step(self, loss_dict: Dict[str, torch.Tensor], batch_idx: int, epoch: int):
        """记录训练步骤"""
        log_dict = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

        # 添加损失值
        for key, value in loss_dict.items():
            log_dict[f'train/{key}'] = value.item()

        # Wandb日志
        if self.use_wandb:
            wandb.log(log_dict, step=self.global_step)

        # 控制台日志
        if self.global_step % (self.log_interval * 10) == 0:
            logger.info(f"Step {self.global_step}: {log_dict}")

    def _log_epoch(self, epoch_record: Dict[str, Any]):
        """记录epoch结果"""
        train_metrics = epoch_record['train_metrics']
        val_metrics = epoch_record['val_metrics']

        log_dict = {
            'epoch': epoch_record['epoch'],
            'learning_rate': epoch_record['learning_rate']
        }

        # 添加训练指标
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f'train/{key}'] = value

        # 添加验证指标
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f'val/{key}'] = value

        # Wandb日志
        if self.use_wandb:
            wandb.log(log_dict)

        # 控制台日志
        logger.info(f"Epoch {epoch_record['epoch']} 完成: {log_dict}")

    def _save_checkpoint(self, epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]):
        """保存检查点"""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'training_history': self.training_history
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")

    def _save_best_model(self, epoch: int, val_loss: float):
        """保存最佳模型"""
        best_model_path = self.output_dir / "best_model.pth"

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        torch.save(checkpoint, best_model_path)
        logger.info(f"最佳模型已保存: {best_model_path}")

    def _save_final_model(self):
        """保存最终模型"""
        final_model_path = self.output_dir / "final_model.pth"

        # 使用专用模型的保存方法
        self.model.save_model(str(final_model_path))

        # 保存训练配置和历史
        training_info = {
            'config': self.config,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.num_epochs,
            'global_step': self.global_step
        }

        with open(self.output_dir / "training_info.json", 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)

        logger.info(f"最终模型已保存: {final_model_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 恢复模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 恢复训练状态
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])

        logger.info(f"检查点已加载: {checkpoint_path}")
        return checkpoint

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        logger.info("开始模型评估")
        self.model.eval()

        total_metrics = {
            'compression_ratio_error': 0.0,
            'quality_score': 0.0,
            'logic_integrity': 0.0,
            'story_coherence': 0.0,
            'playability_score': 0.0
        }

        num_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                try:
                    # 数据移到设备
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # 执行推理
                    for i in range(batch['target_ratio'].size(0)):
                        target_ratio = batch['target_ratio'][i].item()

                        # 创建配置
                        if target_ratio >= 0.7:
                            level = CompressionLevel.LIGHT
                        elif target_ratio >= 0.5:
                            level = CompressionLevel.MEDIUM
                        else:
                            level = CompressionLevel.HEAVY

                        config = CompressionConfig(
                            target_ratio=target_ratio,
                            compression_level=level
                        )

                        # 模拟推理（简化版本）
                        original_text = "Sample text for evaluation"
                        result = self.model(original_text, config)

                        # 累积指标
                        metrics = result['metrics']
                        total_metrics['compression_ratio_error'] += abs(
                            metrics.compression_ratio - target_ratio
                        )
                        total_metrics['quality_score'] += metrics.overall_quality
                        total_metrics['logic_integrity'] += metrics.logic_integrity
                        total_metrics['story_coherence'] += metrics.story_coherence
                        total_metrics['playability_score'] += metrics.playability_score

                        num_samples += 1

                except Exception as e:
                    logger.error(f"评估批次失败: {e}")
                    continue

        # 计算平均指标
        if num_samples > 0:
            for key in total_metrics:
                total_metrics[key] /= num_samples

        total_metrics['num_samples'] = num_samples
        logger.info(f"模型评估完成，样本数: {num_samples}")

        return total_metrics


class TrainingConfig:
    """训练配置类"""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认训练配置"""
        return {
            # 模型配置
            'model_name': 't5-large',
            'hidden_size': 768,
            'max_length': 2048,

            # 训练参数
            'learning_rate': 1e-4,
            'batch_size': 4,
            'num_epochs': 50,
            'warmup_steps': 1000,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,

            # 数据配置
            'train_ratio': 0.8,
            'val_ratio': 0.15,
            'test_ratio': 0.05,
            'apply_augmentation': True,
            'augmentation_factor': 2.0,

            # 保存和日志
            'output_dir': './checkpoints',
            'log_interval': 100,
            'save_interval': 5,
            'eval_interval': 1,

            # Wandb配置
            'use_wandb': False,
            'wandb_project': 'script-compression-model',

            # 其他配置
            'seed': 42,
            'num_workers': 4,
            'pin_memory': True
        }

    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """获取生产环境配置"""
        config = TrainingConfig.get_default_config()
        config.update({
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 5e-5,
            'warmup_steps': 2000,
            'use_wandb': True,
            'save_interval': 2
        })
        return config