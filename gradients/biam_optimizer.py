"""
BIAM Optimizer
论文 Eq.4-6 的一阶近似双层优化器：
  Step1（真实下层更新）: w(t) = w(t-1) - γ_w ∇_w R(θ(t), w(t-1))，样本权重由 ν(·;θ) 以 per-sample loss 为输入
  Step2（虚拟步）:       ŵ(θ)  = w(t-1) - γ_w ∇_w R(θ(t), w(t))，可微
  Step3（上层更新）:     θ(t+1) = θ(t) - γ_θ ∇_θ L_val(ŵ)，超梯度经虚拟步回传
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score


class BIAMOptimizer:
    """
    BIAM 双层优化器（一阶近似）
    """
    
    def __init__(self, config, biam_model, weighting_network):
        """
        Args:
            config: BIAM configuration
            biam_model: BIAMModel（内含 additive_model 与 weighting_network）
            weighting_network: 加权网络 ν(·;θ)
        """
        self.config = config
        self.biam_model = biam_model
        self.additive_model = biam_model.additive_model
        self.weighting_network = weighting_network
        self.device = config.device if hasattr(config, 'device') else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.upper_lr = config.upper_lr
        self.lower_lr = config.lower_lr
        
        # 是否启用双层优化（BIAM-B 消融时置 False：样本权重恒为均匀）
        self.use_bilevel = getattr(config, 'use_bilevel', True)
        
        self._initialize_optimizers()
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'test_metrics': []
        }
    
    def _initialize_optimizers(self):
        """
        下层优化加性模型参数 w，上层优化加权网络参数 θ
        （论文使用一阶方法，SGD 无动量与公式一致）
        BIAM² 门控 γ 使用 lower_lr × pair_gate_lr_scale 的独立参数组：
        真交互的梯度持续且相干，小步长仍能积累；离群标签驱动的"爬行开门"被放缓
        """
        gate_lr_scale = getattr(self.config, 'pair_gate_lr_scale', 1.0)
        base_params, gate_params = [], []
        for name, p in self.additive_model.named_parameters():
            if not p.requires_grad:
                continue
            (gate_params if 'C_gate' in name else base_params).append(p)
        lower_groups = [{'params': base_params, 'lr': self.lower_lr}]
        if gate_params:
            lower_groups.append({'params': gate_params, 'lr': self.lower_lr * gate_lr_scale})
        self.lower_optimizer = optim.SGD(lower_groups)
        self._gate_param_set = set(id(p) for p in gate_params)

        self.upper_optimizer = optim.SGD(
            self.weighting_network.parameters(),
            lr=self.upper_lr
        )

    def _param_lr(self, param):
        """参数对应的学习率（门控参数组用 lower_lr × pair_gate_lr_scale）"""
        return self.lower_lr * getattr(self.config, 'pair_gate_lr_scale', 1.0) \
            if id(param) in self._gate_param_set else self.lower_lr
    
    def _per_sample_loss(self, predictions, target):
        """
        per-sample loss：回归为逐样本 MSE，分类为逐样本交叉熵
        """
        if self.config.task == 'regression':
            if target.dim() == 1 and predictions.dim() == 2 and predictions.shape[1] == 1:
                target = target.unsqueeze(1)
            return F.mse_loss(predictions, target, reduction='none').mean(dim=1, keepdim=True)
        else:
            return F.cross_entropy(predictions, target.long(), reduction='none').unsqueeze(1)
    
    def _nu_input(self, losses):
        """
        加权网络输入：per-sample loss 的批内标准化（z-score）
        损失尺度在不同任务/训练阶段差异巨大，标准化使 ν 的响应范围稳定
        （对应论文附录提到的初始化与缩放约束）
        """
        if losses.numel() > 1 and losses.std() > 1e-8:
            return (losses - losses.mean()) / (losses.std() + 1e-8)
        return losses - losses.mean()
    
    def _weighted_risk(self, losses, weights):
        """
        经验风险 R = (1/n) Σ ν(ℓ_i;θ)·ℓ_i（加权）
        权重按均值归一化（保持相对重加权语义，避免 sigmoid 初值 ~0.5 缩减梯度尺度）
        """
        if weights.numel() > 1:
            weights = weights / (weights.mean() + 1e-8)
        return (weights * losses).mean()
    
    def _bilevel_step(self, x, y, xv, yv):
        """
        单次双层优化步（论文 Eq.4-6）
        """
        add_params = [p for p in self.additive_model.parameters() if p.requires_grad]
        
        if self.use_bilevel:
            # ---------- Step 1: 真实下层更新 ----------
            losses1 = self._per_sample_loss(self.additive_model(x), y)
            # 输入 detach：下层更新时 ν 只作为加权函数，不对 θ 求导
            w1 = self.weighting_network(self._nu_input(losses1.detach()))
            R1 = self._weighted_risk(losses1, w1) + self.additive_model.get_penalty()
            
            self.lower_optimizer.zero_grad()
            R1.backward()
            torch.nn.utils.clip_grad_norm_(add_params, max_norm=1.0)
            self.lower_optimizer.step()
            
            # ---------- Step 2: 虚拟步（可微）----------
            named_params = dict(self.additive_model.named_parameters())
            losses2 = self._per_sample_loss(self.additive_model(x), y)
            # 经 θ 可微：losses2 不 detach，w2 依赖 ν 的参数 θ
            w2 = self.weighting_network(self._nu_input(losses2.detach()))
            R2 = self._weighted_risk(losses2, w2) + self.additive_model.get_penalty()
            
            grads = torch.autograd.grad(R2, add_params, create_graph=True, allow_unused=True)
            virtual_params = {}
            for (name, p), g in zip(named_params.items(), grads):
                if g is not None:
                    # 虚拟步与真实下层更新保持一致的分组学习率
                    virtual_params[name] = p - self._param_lr(p) * g
            
            # ---------- Step 3: 上层更新（超梯度经虚拟步回传）----------
            val_pred = torch.func.functional_call(self.additive_model, virtual_params, (xv,))
            val_losses = self._per_sample_loss(val_pred, yv)
            L_val = val_losses.mean()
            
            self.upper_optimizer.zero_grad()
            self.lower_optimizer.zero_grad()
            L_val.backward()
            torch.nn.utils.clip_grad_norm_(self.weighting_network.parameters(), max_norm=1.0)
            self.upper_optimizer.step()
            
            train_loss = R2.detach().item()
            val_loss = L_val.detach().item()
        else:
            # BIAM-B 消融：固定均匀样本权重，仅做下层更新
            losses1 = self._per_sample_loss(self.additive_model(x), y)
            w1 = torch.ones_like(losses1)
            R1 = self._weighted_risk(losses1, w1) + self.additive_model.get_penalty()
            
            self.lower_optimizer.zero_grad()
            R1.backward()
            torch.nn.utils.clip_grad_norm_(add_params, max_norm=1.0)
            self.lower_optimizer.step()
            
            with torch.no_grad():
                val_loss = self._per_sample_loss(self.additive_model(xv), yv).mean().item()
            train_loss = R1.detach().item()
        
        return train_loss, val_loss
    
    def train_epoch(self, train_loader, val_loader, epoch):
        """
        训练一个 epoch：逐 batch 执行双层优化步
        BIAM² 预热：前 pair_warmup_epochs 冻结成对交互（pair_active=False），
        让双层权重先识别离群/失衡样本，之后门控再开始"保守开门"
        """
        self.biam_model.train()
        self.weighting_network.train()

        warmup = getattr(self.config, 'pair_warmup_epochs', 0)
        if hasattr(self.additive_model, 'pair_active'):
            self.additive_model.pair_active = epoch >= warmup
        
        total_train_loss = 0.0
        total_val_loss = 0.0
        num_batches = 0
        
        # 验证集数据缓存（每个 batch 使用同一验证集）
        val_batches = list(val_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 验证 batch 轮转取用
            val_data, val_target = val_batches[batch_idx % len(val_batches)]
            val_data, val_target = val_data.to(self.device), val_target.to(self.device)
            
            train_loss, val_loss = self._bilevel_step(data, target, val_data, val_target)
            
            total_train_loss += train_loss
            total_val_loss += val_loss
            num_batches += 1
        
        avg_train_loss = total_train_loss / max(num_batches, 1)
        avg_val_loss = total_val_loss / max(num_batches, 1)
        
        self.training_history['train_loss'].append(avg_train_loss)
        self.training_history['val_loss'].append(avg_val_loss)
        
        return {
            'loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
    
    def evaluate(self, test_data):
        """
        在测试集上评估：回归 MSE/RMSE/MAE；分类 Accuracy/Macro-F1（论文指标）/Weighted-F1
        """
        self.biam_model.eval()
        
        X_test, y_test = test_data
        if not torch.is_tensor(X_test):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if not torch.is_tensor(y_test):
            y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32)
        
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        with torch.no_grad():
            predictions = self.additive_model(X_test)
            
            if self.config.task == 'regression':
                if y_test.dim() == 1 and predictions.dim() == 2 and predictions.shape[1] == 1:
                    y_t = y_test.unsqueeze(1)
                else:
                    y_t = y_test
                mse = F.mse_loss(predictions, y_t).item()
                mae = F.l1_loss(predictions, y_t).item()
                return {
                    'mse': mse,
                    'rmse': float(np.sqrt(mse)),
                    'mae': mae
                }
            else:
                pred_labels = predictions.argmax(dim=1).cpu().numpy()
                true_labels = y_test.long().cpu().numpy()
                return {
                    'accuracy': accuracy_score(true_labels, pred_labels),
                    'f1_macro': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
                    'f1_score': f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
                }
    
    def get_training_history(self):
        """获取训练历史"""
        return self.training_history
    
    def save_checkpoint(self, filepath):
        """保存检查点"""
        torch.save({
            'additive_model': self.additive_model.state_dict(),
            'weighting_network': self.weighting_network.state_dict(),
            'upper_optimizer': self.upper_optimizer.state_dict(),
            'lower_optimizer': self.lower_optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.additive_model.load_state_dict(checkpoint['additive_model'])
        self.weighting_network.load_state_dict(checkpoint['weighting_network'])
        self.upper_optimizer.load_state_dict(checkpoint['upper_optimizer'])
        self.lower_optimizer.load_state_dict(checkpoint['lower_optimizer'])
        self.training_history = checkpoint.get('training_history', self.training_history)
