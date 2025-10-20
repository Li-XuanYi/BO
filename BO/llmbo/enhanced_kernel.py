"""
策略2: LLM增强的复合核函数 (带自适应gamma)
考虑参数间的物理耦合关系

修改点:
1. LLMEnhancedKernel添加update_gamma方法
2. 在每次BO迭代后调用update_gamma更新耦合强度
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, Matern
from typing import Dict, Tuple


class LLMEnhancedKernel(Kernel):
    """
    LLM增强的复合核函数 (带自适应gamma)
    
    结构:
    k(theta, theta') = k_base(theta, theta') + gamma * k_coupling(theta, theta')
    
    核心改进:
    - gamma会根据优化历史自适应调整 (论文公式7)
    - 收敛快 -> gamma增大 -> 加强耦合
    - 收敛慢 -> gamma减小 -> 削弱耦合
    """
    
    def __init__(
        self,
        length_scales: np.ndarray = None,
        coupling_matrix: np.ndarray = None,
        coupling_strength: float = 0.3,
        nu: float = 2.5,
        min_gamma: float = 0.01,
        max_gamma: float = 1.0,
        adaptation_rate: float = 0.1
    ):
        """
        初始化LLM增强核函数
        
        参数:
            length_scales: 每个参数的长度尺度
            coupling_matrix: 参数耦合矩阵 (3x3)
            coupling_strength: 耦合项的权重系数 gamma (初始值)
            nu: Matern核的参数
            min_gamma: gamma下限
            max_gamma: gamma上限
            adaptation_rate: gamma调整速率 (论文中为0.1)
        """
        self.length_scales = np.asarray(length_scales) if length_scales is not None else np.ones(3)
        self.coupling_matrix = np.asarray(coupling_matrix) if coupling_matrix is not None else np.eye(3)
        self.coupling_strength = coupling_strength
        self.initial_gamma = coupling_strength  # 保存初始值
        self.nu = nu
        
        # gamma自适应参数
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.adaptation_rate = adaptation_rate
        
        # 历史记录
        self.gamma_history = [coupling_strength]
        self.f_min_history = []
        
        # 基础Matern核
        self.base_kernel = Matern(nu=nu, length_scale=self.length_scales)
    
    def update_gamma(self, current_f_min: float, iteration: int, verbose: bool = True) -> float:
        """
        根据当前最优目标值更新gamma
        
        论文公式(7):
        gamma_k+1 = gamma_k * (1 + 0.1 * (f_min,k - f_min,k-1) / f_min,k-1)
        
        参数:
            current_f_min: 当前迭代的最优目标值
            iteration: 当前迭代次数
            verbose: 是否打印更新信息
            
        返回:
            更新后的gamma值
        """
        # 记录当前f_min
        self.f_min_history.append(current_f_min)
        
        # 第一次迭代无法计算改进率,保持初始gamma
        if iteration == 0 or len(self.f_min_history) < 2:
            if verbose:
                print(f"[Iter {iteration}] Gamma保持初始值: {self.coupling_strength:.4f}")
            return self.coupling_strength
        
        # 获取前一次的f_min
        f_min_prev = self.f_min_history[-2]
        f_min_curr = self.f_min_history[-1]
        
        # 避免除零错误
        if abs(f_min_prev) < 1e-10:
            if verbose:
                print(f"[Iter {iteration}] f_min_prev接近0,gamma保持不变: {self.coupling_strength:.4f}")
            return self.coupling_strength
        
        # 计算改进率 (论文公式7)
        improvement_rate = (f_min_curr - f_min_prev) / abs(f_min_prev)
        
        # 更新gamma
        gamma_old = self.coupling_strength
        gamma_new = gamma_old * (1 + self.adaptation_rate * improvement_rate)
        
        # 限制在[min_gamma, max_gamma]范围内
        gamma_new = np.clip(gamma_new, self.min_gamma, self.max_gamma)
        
        # 更新并记录
        self.coupling_strength = gamma_new
        self.gamma_history.append(gamma_new)
        
        if verbose:
            print(f"[Iter {iteration}] Gamma更新:")
            print(f"  f_min: {f_min_prev:.6f} -> {f_min_curr:.6f}")
            print(f"  改进率: {improvement_rate:.6f}")
            print(f"  gamma: {gamma_old:.4f} -> {gamma_new:.4f}")
            
            # 解释gamma变化的物理意义
            if gamma_new > gamma_old:
                print(f"  含义: 收敛加速,加强参数耦合")
            elif gamma_new < gamma_old:
                print(f"  含义: 收敛减速,削弱耦合避免过拟合")
            else:
                print(f"  含义: gamma已达边界限制")
        
        return gamma_new
    
    def get_gamma_history(self) -> Dict:
        """
        返回gamma和f_min的完整历史
        
        返回:
            包含gamma_history和f_min_history的字典
        """
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
        """
        计算核函数值
        
        使用当前的coupling_strength (gamma)
        """
        # 计算基础核
        if eval_gradient:
            K_base, dK_base = self.base_kernel(X, Y, eval_gradient=True)
        else:
            K_base = self.base_kernel(X, Y)
        
        # 计算耦合项
        K_coupling = self._compute_coupling_kernel(X, Y)
        
        # 复合核 (使用当前的coupling_strength)
        K = K_base + self.coupling_strength * K_coupling
        
        if eval_gradient:
            # 梯度也需要修改 (这里简化处理)
            dK = dK_base  # 目前只返回基础核的梯度
            return K, dK
        else:
            return K
    
    def _compute_coupling_kernel(self, X, Y=None):
        """
        计算耦合项
        
        基于参数间的物理相互作用:
        - current1 <-> charging_number: 强耦合 (0.7)
        - charging_number <-> current2: 中等耦合 (0.6)
        - current1 <-> current2: 弱耦合 (0.3)
        """
        if Y is None:
            Y = X
        
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # 确保输入是2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        n_X = X.shape[0]
        n_Y = Y.shape[0]
        K_coupling = np.zeros((n_X, n_Y))
        
        for i in range(n_X):
            for j in range(n_Y):
                # 参数差异向量
                delta = X[i] - Y[j]
                
                # 耦合效应: (delta_theta)^T * W * (delta_theta)
                coupling_effect = np.dot(delta, np.dot(self.coupling_matrix, delta))
                
                # 高斯型耦合核
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
    """
    获取LLM推荐的核函数配置
    
    这些参数基于论文和物理知识
    """
    config = {
        'length_scales': np.array([
            0.5,    # current1: 较小,因为电流敏感
            3.0,    # charging_number: 较大,因为阶段转换点灵活
            0.8     # current2: 中等,因为第二阶段电流也很重要
        ]),
        
        'coupling_matrix': np.array([
            [1.0, 0.7, 0.3],   # current1 与其他参数的耦合
            [0.7, 1.0, 0.6],   # charging_number 的耦合
            [0.3, 0.6, 1.0]    # current2 的耦合
        ]),
        
        'coupling_strength': 0.3,  # gamma: 耦合项的权重 (初始值)
        
        'reasoning': """
        耦合矩阵说明:
        - current1 <-> charging_number (0.7): 强耦合
          高电流需要早切换以避免过热
        - charging_number <-> current2 (0.6): 中等耦合
          切换点决定了第二阶段的工作量
        - current1 <-> current2 (0.3): 弱耦合
          通过热效应和电压的间接影响
        """
    }
    
    return config


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试LLM增强核函数的自适应gamma功能")
    print("="*70)
    
    # 获取配置
    config = get_llm_kernel_config()
    
    print("\n核函数配置:")
    print("-"*70)
    print(f"Length scales: {config['length_scales']}")
    print(f"Initial coupling strength: {config['coupling_strength']}")
    print(f"Coupling matrix:\n{config['coupling_matrix']}")
    
    # 创建核函数
    kernel = LLMEnhancedKernel(
        length_scales=config['length_scales'],
        coupling_matrix=config['coupling_matrix'],
        coupling_strength=config['coupling_strength']
    )
    
    # 测试gamma自适应更新
    print("\n\n测试场景: 模拟优化过程中gamma的自适应调整")
    print("-"*70)
    
    # 模拟快速收敛场景
    print("\n场景1: 快速收敛 (目标值快速下降)")
    f_min_sequence = [1.0, 0.8, 0.5, 0.3, 0.2, 0.15]
    
    for i, f_min in enumerate(f_min_sequence):
        gamma = kernel.update_gamma(f_min, iteration=i, verbose=True)
        print()
    
    # 查看历史
    history = kernel.get_gamma_history()
    print("\n历史记录:")
    print(f"Gamma历史: {[f'{g:.4f}' for g in history['gamma_history']]}")
    print(f"f_min历史: {[f'{f:.4f}' for f in history['f_min_history']]}")
    
    # 重置并测试慢收敛
    kernel.reset_gamma()
    print("\n\n场景2: 慢收敛 (目标值缓慢下降)")
    print("-"*70)
    f_min_sequence = [1.0, 0.95, 0.92, 0.90, 0.89, 0.88]
    
    for i, f_min in enumerate(f_min_sequence):
        gamma = kernel.update_gamma(f_min, iteration=i, verbose=True)
        print()
    
    history = kernel.get_gamma_history()
    print("\n历史记录:")
    print(f"Gamma历史: {[f'{g:.4f}' for g in history['gamma_history']]}")
    print(f"f_min历史: {[f'{f:.4f}' for f in history['f_min_history']]}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)