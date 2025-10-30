"""
策略3: LLM增强的候选采样 - 加权采集函数

实现论文公式(8)和(9):
- α_EI^LLM(θ) = EI(θ) * W_LLM(θ)
- W_LLM(θ) = ∏ N(θ_j | μ_j, σ_j²)

LLM提供μ和σ，用于引导采样到高潜力区域
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import warnings


class LLMWeightedAcquisition:
    """
    LLM加权的期望改进采集函数
    
    结合标准EI和LLM引导的权重函数
    """
    
    def __init__(
        self,
        pbounds: Dict,
        param_names: List[str],
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None
    ):
        """
        初始化加权采集函数
        
        参数:
            pbounds: 参数边界字典
            param_names: 参数名称列表（有序）
            mu: LLM建议的参数均值（默认为参数范围中点）
            sigma: LLM建议的参数标准差（默认为参数范围的1/6）
        """
        self.pbounds = pbounds
        self.param_names = param_names
        
        # 初始化μ和σ
        if mu is None:
            # 默认：参数范围的中点
            mu = np.array([
                (pbounds[name][0] + pbounds[name][1]) / 2.0
                for name in param_names
            ])
        
        if sigma is None:
            # 默认：参数范围的1/6（覆盖约99.7%的范围）
            sigma = np.array([
                (pbounds[name][1] - pbounds[name][0]) / 6.0
                for name in param_names
            ])
        
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)
        
        print(f"\n[LLM加权采集函数] 初始化完成")
        print(f"  μ (建议中心): {self.mu}")
        print(f"  σ (建议范围): {self.sigma}")
    
    def update_distribution(
        self,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None
    ):
        """
        更新LLM建议的分布参数
        
        参数:
            mu: 新的均值
            sigma: 新的标准差
        """
        if mu is not None:
            self.mu = np.asarray(mu)
        if sigma is not None:
            self.sigma = np.asarray(sigma)
        
        print(f"\n[LLM加权采集函数] 分布已更新")
        print(f"  新μ: {self.mu}")
        print(f"  新σ: {self.sigma}")
    
    def _compute_w_llm(self, theta: np.ndarray) -> float:
        """
        计算LLM权重函数 W_LLM(θ)
        
        论文公式(9):
        W_LLM = ∏ (1/√(2πσ_j²)) exp(-(θ_j - μ_j)²/(2σ_j²))
        
        参数:
            theta: 参数向量 (shape: (n_params,))
            
        返回:
            权重值
        """
        # 确保输入是1D
        theta = np.atleast_1d(theta)
        
        # 计算每个参数的高斯权重
        weights = norm.pdf(theta, loc=self.mu, scale=self.sigma)
        
        # 乘积（论文公式(9)）
        w_llm = np.prod(weights)
        
        return w_llm
    
    def _compute_ei(
        self,
        theta: np.ndarray,
        gp,
        f_max: float,
        xi: float = 0.01
    ) -> float:
        """
        计算标准期望改进 EI(θ)
        
        参数:
            theta: 参数向量
            gp: 高斯过程回归器
            f_max: 当前最优目标值
            xi: 探索参数
            
        返回:
            EI值
        """
        # 确保输入形状正确
        theta = np.atleast_2d(theta)
        
        # GP预测
        mu, sigma = gp.predict(theta, return_std=True)
        
        # 处理返回值：可能是标量或数组
        mu = float(mu) if np.ndim(mu) == 0 else float(mu.flatten()[0])
        sigma = float(sigma) if np.ndim(sigma) == 0 else float(sigma.flatten()[0])
        
        # 避免除零
        if sigma < 1e-10:
            return 0.0
        
        # 标准化改进
        z = (mu - f_max - xi) / sigma
        
        # EI计算
        ei = (mu - f_max - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return max(0.0, ei)
    
    def compute(
        self,
        theta: np.ndarray,
        gp,
        f_max: float,
        xi: float = 0.01
    ) -> float:
        """
        计算LLM加权的期望改进
        
        论文公式(8):
        α_EI^LLM(θ) = EI(θ) * W_LLM(θ)
        
        参数:
            theta: 参数向量
            gp: 高斯过程回归器
            f_max: 当前最优目标值
            xi: 探索参数
            
        返回:
            加权EI值
        """
        # 计算标准EI
        ei = self._compute_ei(theta, gp, f_max, xi)
        
        # 计算LLM权重
        w_llm = self._compute_w_llm(theta.flatten())
        
        # 加权EI（论文公式(8)）
        weighted_ei = ei * w_llm
        
        return weighted_ei
    
    def suggest(
        self,
        gp,
        f_max: float,
        n_random: int = 10000,
        n_l_bfgs_b: int = 10,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        建议下一个采样点
        
        策略:
        1. 随机采样n_random个点，选择最优的作为起点
        2. 使用L-BFGS-B优化n_l_bfgs_b次
        3. 返回最优点
        
        参数:
            gp: 高斯过程回归器
            f_max: 当前最优目标值
            n_random: 随机采样数量
            n_l_bfgs_b: L-BFGS-B优化次数
            xi: 探索参数
            
        返回:
            建议的参数向量
        """
        # 参数边界
        bounds = np.array([self.pbounds[name] for name in self.param_names])
        
        # 1. 随机采样
        random_samples = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(n_random, len(self.param_names))
        )
        
        # 评估随机样本
        random_acq = np.array([
            self.compute(sample, gp, f_max, xi)
            for sample in random_samples
        ])
        
        # 最优随机点
        best_random_idx = np.argmax(random_acq)
        x_best = random_samples[best_random_idx]
        acq_best = random_acq[best_random_idx]
        
        # 2. L-BFGS-B优化
        for _ in range(n_l_bfgs_b):
            # 随机起点
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            
            # 定义目标函数（最小化负的加权EI）
            def objective(theta):
                return -self.compute(theta, gp, f_max, xi)
            
            # 优化
            try:
                res = minimize(
                    objective,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                # 更新最优解
                acq_opt = -res.fun
                if acq_opt > acq_best:
                    x_best = res.x
                    acq_best = acq_opt
            except:
                continue
        
        return x_best


def parse_llm_guidance_to_distribution(
    guidance: Dict,
    param_names: List[str],
    pbounds: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将LLM的guidance解析为μ和σ
    
    策略:
    1. 如果guidance包含recommended_focus，使用它来设置μ和σ
    2. 否则，根据strategy调整σ（探索时σ大，利用时σ小）
    
    参数:
        guidance: LLM返回的指导字典
        param_names: 参数名称列表
        pbounds: 参数边界
        
    返回:
        (mu, sigma) 元组
    """
    n_params = len(param_names)
    
    # 默认值
    mu_default = np.array([
        (pbounds[name][0] + pbounds[name][1]) / 2.0
        for name in param_names
    ])
    sigma_default = np.array([
        (pbounds[name][1] - pbounds[name][0]) / 6.0
        for name in param_names
    ])
    
    # 尝试从guidance提取信息
    if guidance is None:
        return mu_default, sigma_default
    
    # 方法1: 使用recommended_focus
    if 'recommended_focus' in guidance and guidance['recommended_focus']:
        try:
            mu = []
            sigma = []
            
            for name in param_names:
                if name in guidance['recommended_focus']:
                    focus = guidance['recommended_focus'][name]
                    
                    # 计算μ（范围中点）
                    mu_param = (focus['min'] + focus['max']) / 2.0
                    
                    # 计算σ（范围的1/4，比默认更集中）
                    sigma_param = (focus['max'] - focus['min']) / 4.0
                    
                    mu.append(mu_param)
                    sigma.append(sigma_param)
                else:
                    # 使用默认值
                    mu.append(mu_default[param_names.index(name)])
                    sigma.append(sigma_default[param_names.index(name)])
            
            return np.array(mu), np.array(sigma)
        except:
            pass
    
    # 方法2: 根据strategy调整σ
    strategy = guidance.get('strategy', 'balanced')
    
    if strategy == 'exploration':
        # 探索：σ更大
        sigma = sigma_default * 1.5
    elif strategy == 'exploitation':
        # 利用：σ更小，μ偏向当前最优
        sigma = sigma_default * 0.5
        # TODO: 可以进一步调整μ向当前最优点偏移
    else:
        # 平衡
        sigma = sigma_default
    
    return mu_default, sigma


# 测试代码
if __name__ == "__main__":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    
    print("="*70)
    print("测试LLM加权采集函数")
    print("="*70)
    
    # 模拟问题设置
    pbounds = {
        'current1': (3.0, 6.0),
        'charging_number': (5.0, 25.0),
        'current2': (1.0, 4.0)
    }
    param_names = ['current1', 'charging_number', 'current2']
    
    # 创建一些训练数据
    np.random.seed(42)
    X_train = np.array([
        [3.5, 10.0, 2.0],
        [4.0, 15.0, 2.5],
        [5.0, 12.0, 3.0],
        [4.5, 18.0, 1.5],
        [5.5, 8.0, 3.5]
    ])
    y_train = np.array([-1200, -1100, -1250, -1150, -1180])
    
    # 训练GP
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
    gp.fit(X_train, y_train)
    
    f_max = np.max(y_train)
    
    # 测试1: 默认分布
    print("\n测试1: 使用默认分布")
    print("-"*70)
    acq = LLMWeightedAcquisition(pbounds, param_names)
    
    # 测试几个点
    test_points = [
        np.array([4.5, 15.0, 2.5]),  # 中心点
        np.array([3.0, 5.0, 1.0]),   # 边界点
        np.array([6.0, 25.0, 4.0])   # 另一边界
    ]
    
    for i, point in enumerate(test_points):
        ei_llm = acq.compute(point, gp, f_max)
        print(f"点{i+1} {point}: α_EI^LLM = {ei_llm:.6f}")
    
    # 测试2: 建议下一个点
    print("\n测试2: 建议下一个采样点")
    print("-"*70)
    suggested = acq.suggest(gp, f_max, n_random=1000, n_l_bfgs_b=5)
    print(f"建议的点: {suggested}")
    print(f"采集函数值: {acq.compute(suggested, gp, f_max):.6f}")
    
    # 测试3: 更新分布（模拟LLM给出新的建议）
    print("\n测试3: 更新LLM分布")
    print("-"*70)
    # 模拟LLM建议集中在高电流区域
    mu_new = np.array([5.0, 12.0, 2.5])
    sigma_new = np.array([0.5, 3.0, 0.5])
    acq.update_distribution(mu_new, sigma_new)
    
    suggested_new = acq.suggest(gp, f_max, n_random=1000, n_l_bfgs_b=5)
    print(f"新建议的点: {suggested_new}")
    print(f"采集函数值: {acq.compute(suggested_new, gp, f_max):.6f}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)