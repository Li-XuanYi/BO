"""
策略2: LLM增强的复合核函数 (集成LLM智能gamma调整)

增强内容:
1. 保留原有公式调整 (论文公式7)
2. 每5次迭代咨询LLM获取智能建议
3. 结合公式和LLM建议做最终决策
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Matern
from typing import Dict, Tuple, Optional, List


class LLMEnhancedKernel(Kernel):
    """
    LLM增强的复合核函数 (带智能gamma调整)
    
    调整策略:
    1. 每次迭代: 使用公式调整 (快速响应)
    2. 每5次迭代: 咨询LLM (战略指导)
    3. 结合两者: LLM建议作为修正因子
    """
    
    def __init__(
        self,
        length_scales: np.ndarray = None,
        coupling_matrix: np.ndarray = None,
        coupling_strength: float = 0.3,
        nu: float = 2.5,
        min_gamma: float = 0.01,
        max_gamma: float = 1.0,
        adaptation_rate: float = 0.1,
        use_llm_guidance: bool = True,
        llm_model: str = "gpt-3.5-turbo"
    ):
        """
        初始化LLM增强核函数
        
        参数:
            use_llm_guidance: 是否使用LLM智能指导
            llm_model: LLM模型名称
        """
        self.length_scales = np.asarray(length_scales) if length_scales is not None else np.ones(3)
        self.coupling_matrix = np.asarray(coupling_matrix) if coupling_matrix is not None else np.eye(3)
        self.coupling_strength = coupling_strength
        self.initial_gamma = coupling_strength
        self.nu = nu
        
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.adaptation_rate = adaptation_rate
        self.use_llm_guidance = use_llm_guidance
        self.llm_model = llm_model
        
        self.gamma_history = [coupling_strength]
        self.f_min_history = []
        
        self.base_kernel = Matern(nu=nu, length_scale=self.length_scales)
        
        self.llm_advisor = None
        if use_llm_guidance:
            try:
                import sys
                import os
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sys.path.insert(0, parent_dir)
                from llmbo.llm_gamma_advisor import LLMGammaAdvisor
                self.llm_advisor = LLMGammaAdvisor(llm_model=llm_model)
                print(f"[LLMEnhancedKernel] LLM智能调整已启用 (模型: {llm_model})")
            except Exception as e:
                print(f"[LLMEnhancedKernel] 警告: 无法加载LLM顾问 - {e}")
                self.use_llm_guidance = False
    
    def update_gamma(
        self,
        current_f_min: float,
        iteration: int,
        optimization_history: Optional[List[Dict]] = None,
        verbose: bool = True
    ) -> float:
        """
        更新gamma (公式 + LLM智能调整)
        
        参数:
            current_f_min: 当前最优目标值
            iteration: 当前迭代次数
            optimization_history: 完整优化历史 (供LLM分析)
            verbose: 是否打印更新信息
            
        返回:
            更新后的gamma值
        """
        self.f_min_history.append(current_f_min)
        
        if iteration == 0 or len(self.f_min_history) < 2:
            if verbose:
                print(f"[Iter {iteration}] Gamma保持初始值: {self.coupling_strength:.4f}")
            return self.coupling_strength
        
        f_min_prev = self.f_min_history[-2]
        f_min_curr = self.f_min_history[-1]
        
        if abs(f_min_prev) < 1e-10:
            if verbose:
                print(f"[Iter {iteration}] f_min_prev接近0,gamma保持不变: {self.coupling_strength:.4f}")
            return self.coupling_strength
        
        gamma_old = self.coupling_strength
        
        improvement_rate = (f_min_curr - f_min_prev) / abs(f_min_prev)
        gamma_formula = gamma_old * (1 + self.adaptation_rate * improvement_rate)
        gamma_formula = np.clip(gamma_formula, self.min_gamma, self.max_gamma)
        
        should_consult_llm = (
            self.use_llm_guidance and 
            self.llm_advisor is not None and 
            optimization_history is not None and
            self.llm_advisor.should_consult_llm(iteration)
        )
        
        if should_consult_llm:
            if verbose:
                print(f"\n[Iter {iteration}] 咨询LLM智能顾问...")
            
            llm_advice = self.llm_advisor.get_gamma_recommendation(
                optimization_history=optimization_history,
                current_gamma=gamma_old,
                gamma_history=self.gamma_history,
                f_min_history=self.f_min_history
            )
            
            gamma_llm = llm_advice.get('recommended_gamma', gamma_formula)
            confidence = llm_advice.get('confidence', 'medium')
            
            if confidence == 'high':
                weight_llm = 0.7
            elif confidence == 'medium':
                weight_llm = 0.5
            else:
                weight_llm = 0.3
            
            gamma_new = weight_llm * gamma_llm + (1 - weight_llm) * gamma_formula
            gamma_new = np.clip(gamma_new, self.min_gamma, self.max_gamma)
            
            if verbose:
                print(f"[Iter {iteration}] Gamma智能调整:")
                print(f"  公式建议: {gamma_formula:.4f}")
                print(f"  LLM建议: {gamma_llm:.4f} (置信度: {confidence})")
                print(f"  最终gamma: {gamma_new:.4f} (加权融合)")
                print(f"  LLM推理: {llm_advice.get('reasoning', 'N/A')[:80]}...")
        else:
            gamma_new = gamma_formula
            
            if verbose:
                print(f"[Iter {iteration}] Gamma更新:")
                print(f"  f_min: {f_min_prev:.6f} -> {f_min_curr:.6f}")
                print(f"  改进率: {improvement_rate:.6f}")
                print(f"  gamma: {gamma_old:.4f} -> {gamma_new:.4f}")
                
                if gamma_new > gamma_old:
                    print(f"  含义: 收敛加速,加强参数耦合")
                elif gamma_new < gamma_old:
                    print(f"  含义: 收敛减速,削弱耦合避免过拟合")
                else:
                    print(f"  含义: gamma已达边界限制")
        
        self.coupling_strength = gamma_new
        self.gamma_history.append(gamma_new)
        
        return gamma_new
    
    def get_gamma_history(self) -> Dict:
        """返回gamma和f_min的完整历史"""
        return {
            'gamma_history': self.gamma_history,
            'f_min_history': self.f_min_history
        }
    
    def reset_gamma(self):
        """重置gamma到初始值"""
        self.coupling_strength = self.initial_gamma
        self.gamma_history = [self.initial_gamma]
        self.f_min_history = []
        print(f"[LLMEnhancedKernel] Gamma已重置为: {self.initial_gamma:.4f}")
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """计算核函数值"""
        if eval_gradient:
            K_base, dK_base = self.base_kernel(X, Y, eval_gradient=True)
        else:
            K_base = self.base_kernel(X, Y)
        
        K_coupling = self._compute_coupling_kernel(X, Y)
        K = K_base + self.coupling_strength * K_coupling
        
        if eval_gradient:
            dK = dK_base
            return K, dK
        else:
            return K
    
    def _compute_coupling_kernel(self, X, Y=None):
        """计算耦合项"""
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
        
        for i in range(n_X):
            for j in range(n_Y):
                delta = X[i] - Y[j]
                coupling_effect = np.dot(delta, np.dot(self.coupling_matrix, delta))
                K_coupling[i, j] = np.exp(-coupling_effect / 10.0)
        
        return K_coupling
    
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


def get_llm_kernel_config() -> Dict:
    """获取LLM推荐的核函数配置"""
    config = {
        'length_scales': np.array([0.5, 3.0, 0.8]),
        'coupling_matrix': np.array([
            [1.0, 0.7, 0.3],
            [0.7, 1.0, 0.6],
            [0.3, 0.6, 1.0]
        ]),
        'coupling_strength': 0.3,
        'reasoning': """
        耦合矩阵说明:
        - current1 <-> charging_number (0.7): 强耦合
        - charging_number <-> current2 (0.6): 中等耦合
        - current1 <-> current2 (0.3): 弱耦合
        """
    }
    
    return config