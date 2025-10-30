"""
策略2: LLM增强的复合核函数 V3 - 回归核心创新

关键改进:
1. 使用启发式电压梯度（基于物理知识），而非实时SPM计算
2. 避免计算瓶颈，保持BO效率
3. 核心思想：根据论文，耦合项体现参数间的物理相关性，而非精确梯度

论文方程5: k(θ, θ') = Base_Kernel + γ · LLM_coupling
论文方程6: LLM_coupling = Σ w_ij · (∂U/∂θ_i) · (∂U/∂θ_j)
论文方程7: γ_{k+1} = γ_k · (1 + 0.1 · improvement_rate)

设计理念：
- 电压梯度使用启发式值（基于电化学领域知识）
- 重点是LLM智能调整耦合强度γ和权重矩阵W
- 保持计算效率，避免每次kernel评估都调用SPM
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Matern
from typing import Dict, Optional
import uuid


class LLMEnhancedKernelV3(Kernel):
    """
    LLM增强的复合核函数 V3
    
    核心创新：
    1. 启发式电压梯度（避免实时计算）
    2. LLM智能调整耦合强度γ
    3. 保持计算效率
    """
    
    # 类变量：全局共享的LLM advisor
    _global_llm_advisor = None
    _advisor_lock = False
    
    # 类变量：全局共享的gamma和f_min历史
    _global_gamma_history = []
    _global_f_min_history = []
    
    # 启发式电压梯度（基于电化学领域知识）
    # 这些值反映参数对电压的敏感度，来自物理直觉和实验经验
    _heuristic_voltage_gradients = {
        'current1': 2.0,       # 第一阶段电流对电压影响较大
        'charging_number': 0.5,  # 阶段切换点影响中等
        'current2': 1.5        # 第二阶段电流影响中等
    }
    
    def __init__(
        self,
        param_names: tuple = ('current1', 'charging_number', 'current2'),
        length_scales: np.ndarray = None,
        coupling_matrix: np.ndarray = None,
        coupling_strength: float = 0.3,
        nu: float = 2.5,
        min_gamma: float = 0.01,
        max_gamma: float = 1.0,
        use_llm_guidance: bool = True,
        llm_model: str = "gpt-3.5-turbo"
    ):
        """
        初始化LLM增强核函数V3
        
        参数:
            param_names: 参数名称元组（用于获取启发式梯度）
            length_scales: 核函数长度尺度
            coupling_matrix: 耦合矩阵W (w_ij)
            coupling_strength: 耦合强度γ
            nu: Matern核的平滑参数
            min_gamma/max_gamma: γ的范围
            use_llm_guidance: 是否使用LLM指导
            llm_model: LLM模型名称
        """
        self.param_names = param_names
        self.length_scales = np.asarray(length_scales) if length_scales is not None else np.ones(len(param_names))
        self.coupling_matrix = np.asarray(coupling_matrix) if coupling_matrix is not None else np.eye(len(param_names))
        self.coupling_strength = coupling_strength
        self.initial_gamma = coupling_strength
        self.nu = nu
        
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.use_llm_guidance = use_llm_guidance
        self.llm_model = llm_model
        
        # 实例ID（用于识别主kernel）
        self.instance_id = str(uuid.uuid4())
        
        # 基础核函数
        self.base_kernel = Matern(nu=nu, length_scale=self.length_scales)
        
        # 获取启发式电压梯度
        self.voltage_gradients = np.array([
            self._heuristic_voltage_gradients.get(name, 1.0) 
            for name in param_names
        ])
        
        # 初始化LLM advisor（只在第一次创建时）
        if use_llm_guidance and LLMEnhancedKernelV3._global_llm_advisor is None and not LLMEnhancedKernelV3._advisor_lock:
            LLMEnhancedKernelV3._advisor_lock = True
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from llm_advisor import LLMKernelAdvisor
                LLMEnhancedKernelV3._global_llm_advisor = LLMKernelAdvisor(model_name=llm_model)
                print(f"[LLMEnhancedKernelV3] LLM Advisor初始化成功")
            except Exception as e:
                print(f"[LLMEnhancedKernelV3] 警告: LLM Advisor初始化失败 - {e}")
                LLMEnhancedKernelV3._global_llm_advisor = None
            finally:
                LLMEnhancedKernelV3._advisor_lock = False
        
        # 初始化gamma历史
        if len(LLMEnhancedKernelV3._global_gamma_history) == 0:
            LLMEnhancedKernelV3._global_gamma_history = [coupling_strength]
        
        print(f"[LLMEnhancedKernelV3] 初始化完成")
        print(f"  参数: {param_names}")
        print(f"  启发式电压梯度: {self.voltage_gradients}")
        print(f"  初始γ: {coupling_strength:.4f}")
        print(f"  耦合矩阵形状: {self.coupling_matrix.shape}")
    
    def _compute_coupling_kernel(self, X, Y=None):
        """
        计算耦合项（论文方程6）
        
        LLM_coupling = Σ w_ij · (∂U/∂θ_i) · (∂U/∂θ_j)
        
        使用启发式电压梯度，避免实时计算
        """
        if Y is None:
            Y = X
        
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        n_X = X.shape[0]
        n_Y = Y.shape[0]
        K_coupling = np.zeros((n_X, n_Y))
        
        # 使用启发式梯度计算耦合项
        for i in range(n_X):
            for j in range(n_Y):
                # 参数差异
                delta = X[i] - Y[j]
                
                # 计算耦合效应：delta^T · W · delta
                # 其中W是耦合矩阵，反映参数间的物理相关性
                coupling_effect = 0.0
                for p in range(len(delta)):
                    for q in range(len(delta)):
                        # w_ij · grad_i · grad_j · delta_i · delta_j
                        coupling_effect += (
                            self.coupling_matrix[p, q] * 
                            self.voltage_gradients[p] * 
                            self.voltage_gradients[q] * 
                            delta[p] * delta[q]
                        )
                
                # 归一化并转换为核函数值
                K_coupling[i, j] = np.exp(-abs(coupling_effect) / 10.0)
        
        return K_coupling
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        计算核函数值（论文方程5）
        
        k(θ, θ') = Base_Kernel + γ · LLM_coupling
        """
        # 计算基础核
        if eval_gradient:
            K_base, dK_base = self.base_kernel(X, Y, eval_gradient=True)
        else:
            K_base = self.base_kernel(X, Y)
        
        # 计算耦合核
        K_coupling = self._compute_coupling_kernel(X, Y)
        
        # 复合核 = 基础核 + γ * 耦合核
        K = K_base + self.coupling_strength * K_coupling
        
        if eval_gradient:
            # 简化：梯度仅来自基础核
            return K, dK_base
        else:
            return K
    
    def smart_update_gamma(
        self,
        iteration: int,
        f_min_prev: float,
        f_min_curr: float,
        historical_results: list,
        verbose: bool = True
    ) -> float:
        """
        智能更新耦合强度γ（论文方程7）
        
        结合公式和LLM两种方式：
        1. 公式方式：γ_{k+1} = γ_k · (1 + 0.1 · improvement_rate)
        2. LLM方式：分析优化历史，给出调整建议
        
        参数:
            iteration: 当前迭代次数
            f_min_prev: 上一次的最优值
            f_min_curr: 当前的最优值
            historical_results: 历史优化结果
            verbose: 是否打印详细信息
        
        返回:
            更新后的γ值
        """
        gamma_old = self.coupling_strength
        
        # 记录f_min历史
        LLMEnhancedKernelV3._global_f_min_history.append(f_min_curr)
        
        # 计算改进率
        if abs(f_min_prev) > 1e-10:
            improvement_rate = (f_min_prev - f_min_curr) / abs(f_min_prev)
        else:
            improvement_rate = 0.0
        
        # 方法1：公式调整（论文方程7）
        gamma_formula = gamma_old * (1.0 + 0.1 * improvement_rate)
        gamma_formula = np.clip(gamma_formula, self.min_gamma, self.max_gamma)
        
        # 方法2：LLM智能调整（每3次迭代）
        should_use_llm = (
            self.use_llm_guidance and 
            LLMEnhancedKernelV3._global_llm_advisor is not None and
            iteration > 0 and 
            (iteration + 1) % 3 == 0 and
            len(historical_results) >= 3
        )
        
        if should_use_llm:
            try:
                # 让LLM分析优化历史并给出γ调整建议
                llm_advice = LLMEnhancedKernelV3._global_llm_advisor.analyze_gamma_adjustment(
                    current_gamma=gamma_old,
                    improvement_rate=improvement_rate,
                    historical_results=historical_results[-10:],  # 最近10次结果
                    gamma_history=LLMEnhancedKernelV3._global_gamma_history[-10:],
                    f_min_history=LLMEnhancedKernelV3._global_f_min_history[-10:]
                )
                
                gamma_llm = llm_advice['recommended_gamma']
                confidence = llm_advice['confidence']
                
                # 根据置信度加权融合
                confidence_weights = {'high': 0.8, 'medium': 0.7, 'low': 0.5}
                weight_llm = confidence_weights.get(confidence, 0.5)
                
                gamma_new = weight_llm * gamma_llm + (1 - weight_llm) * gamma_formula
                gamma_new = np.clip(gamma_new, self.min_gamma, self.max_gamma)
                
                if verbose:
                    print(f"\n[Iter {iteration}] 智能Gamma调整:")
                    print(f"  公式建议: {gamma_formula:.4f}")
                    print(f"  LLM建议: {gamma_llm:.4f} (置信度: {confidence})")
                    print(f"  最终γ: {gamma_new:.4f}")
                    print(f"  LLM推理: {llm_advice.get('reasoning', 'N/A')[:80]}...")
                
            except Exception as e:
                print(f"  警告: LLM分析失败 - {e}, 使用公式方法")
                gamma_new = gamma_formula
        else:
            gamma_new = gamma_formula
            
            if verbose and (iteration + 1) % 5 == 0:
                print(f"\n[Iter {iteration}] Gamma更新（公式）:")
                print(f"  改进率: {improvement_rate:.6f}")
                print(f"  γ: {gamma_old:.4f} → {gamma_new:.4f}")
        
        # 更新全局状态
        self.coupling_strength = gamma_new
        LLMEnhancedKernelV3._global_gamma_history.append(gamma_new)
        
        return gamma_new
    
    def get_gamma_history(self) -> Dict:
        """返回gamma和f_min的完整历史"""
        return {
            'gamma_history': LLMEnhancedKernelV3._global_gamma_history.copy(),
            'f_min_history': LLMEnhancedKernelV3._global_f_min_history.copy()
        }
    
    def reset_gamma(self):
        """重置gamma到初始值"""
        self.coupling_strength = self.initial_gamma
        LLMEnhancedKernelV3._global_gamma_history = [self.initial_gamma]
        LLMEnhancedKernelV3._global_f_min_history = []
        print(f"[LLMEnhancedKernelV3] Gamma已重置为: {self.initial_gamma:.4f}")
    
    def diag(self, X):
        """计算对角线元素"""
        return np.diag(self(X))
    
    def is_stationary(self):
        """核函数是否平稳"""
        return True
    
    @property
    def requires_vector_input(self):
        """是否需要向量输入"""
        return True
    
    def clone_with_theta(self, theta):
        """克隆kernel（sklearn要求）"""
        # 创建新实例，共享全局状态
        cloned = LLMEnhancedKernelV3(
            param_names=self.param_names,
            length_scales=self.length_scales.copy(),
            coupling_matrix=self.coupling_matrix.copy(),
            coupling_strength=self.coupling_strength,
            nu=self.nu,
            min_gamma=self.min_gamma,
            max_gamma=self.max_gamma,
            use_llm_guidance=self.use_llm_guidance,
            llm_model=self.llm_model
        )
        return cloned


def get_llm_kernel_config_v3(param_names: tuple = ('current1', 'charging_number', 'current2')) -> Dict:
    """
    获取LLM推荐的核函数配置V3
    
    基于电化学领域知识的启发式配置
    """
    n_params = len(param_names)
    
    config = {
        'param_names': param_names,
        'length_scales': np.array([0.5, 3.0, 0.8])[:n_params],  # 根据参数敏感度设定
        'coupling_matrix': np.array([
            [1.0, 0.7, 0.3],  # current1与其他参数的耦合
            [0.7, 1.0, 0.6],  # charging_number与其他参数的耦合
            [0.3, 0.6, 1.0]   # current2与其他参数的耦合
        ])[:n_params, :n_params],
        'coupling_strength': 0.3,
        'reasoning': """
        耦合矩阵说明（基于电化学物理）:
        - current1 ↔ charging_number (0.7): 强耦合
          第一阶段电流和持续时间直接影响第二阶段的起始SOC
        - charging_number ↔ current2 (0.6): 中等耦合
          切换点影响第二阶段的充电效果
        - current1 ↔ current2 (0.3): 弱耦合
          两阶段电流相对独立，但通过电池状态间接耦合
        
        长度尺度说明:
        - current1 (0.5): 短尺度，对电压敏感
        - charging_number (3.0): 长尺度，离散变化较平缓
        - current2 (0.8): 中等尺度
        """
    }
    
    return config


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试LLM增强核函数V3")
    print("="*70)
    
    # 获取配置
    config = get_llm_kernel_config_v3()
    
    # 创建kernel
    kernel = LLMEnhancedKernelV3(
        param_names=config['param_names'],
        length_scales=config['length_scales'],
        coupling_matrix=config['coupling_matrix'],
        coupling_strength=config['coupling_strength'],
        use_llm_guidance=False  # 测试时不使用LLM
    )
    
    # 测试kernel计算
    X1 = np.array([[4.0, 10.0, 2.0]])
    X2 = np.array([[4.5, 15.0, 2.5]])
    
    print("\n测试1: 核函数计算")
    print("-"*70)
    K = kernel(X1, X2)
    print(f"K(X1, X2) = {K[0, 0]:.6f}")
    
    print("\n测试2: Gamma自适应调整")
    print("-"*70)
    fake_history = [
        {'params': {'current1': 4.0, 'charging_number': 10.0, 'current2': 2.0}, 'target': -100.0},
        {'params': {'current1': 4.5, 'charging_number': 15.0, 'current2': 2.5}, 'target': -95.0},
        {'params': {'current1': 5.0, 'charging_number': 12.0, 'current2': 2.2}, 'target': -90.0},
    ]
    
    gamma_new = kernel.smart_update_gamma(
        iteration=2,
        f_min_prev=-95.0,
        f_min_curr=-90.0,
        historical_results=fake_history,
        verbose=True
    )
    
    print(f"\nGamma更新: 0.3 → {gamma_new:.4f}")
    
    print("\n"+ "="*70)
    print("LLM增强核函数V3测试完成")
    print("="*70)
