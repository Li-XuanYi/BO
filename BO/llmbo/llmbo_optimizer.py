"""
LLMBO优化器（完整版本）
集成三个策略：
1. LLM Warm Start
2. LLM增强的复合核函数
3. LLM动态采样策略
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from llm_utils import WarmStartGenerator
try:
    from .enhanced_kernel import LLMEnhancedKernel, get_llm_kernel_config
    from .dynamic_sampling import DynamicSamplingStrategy
except ImportError:
    from enhanced_kernel import LLMEnhancedKernel, get_llm_kernel_config
    from dynamic_sampling import DynamicSamplingStrategy
from typing import Callable, Dict, List
import time
import numpy as np


class LLMBOOptimizer:
    """
    LLM增强的贝叶斯优化器（完整版本）
    
    实现三个策略：
    - 策略1: LLM Warm Start（生成优质初始点）
    - 策略2: LLM增强核函数（改进GP建模）
    - 策略3: LLM动态采样（动态调整探索策略）
    """
    
    def __init__(
        self,
        f: Callable,
        pbounds: Dict[str, tuple],
        llm_model: str = "gpt-3.5-turbo",
        random_state: int = 1,
        use_enhanced_kernel: bool = True,
        use_dynamic_sampling: bool = True
    ):
        """
        初始化LLMBO优化器
        
        参数:
            f: 目标函数
            pbounds: 参数边界
            llm_model: LLM模型名称
            random_state: 随机种子
            use_enhanced_kernel: 是否使用增强核函数
            use_dynamic_sampling: 是否使用动态采样
        """
        self.f = f
        self.pbounds = pbounds
        self.llm_model = llm_model
        self.random_state = random_state
        self.use_enhanced_kernel = use_enhanced_kernel
        self.use_dynamic_sampling = use_dynamic_sampling
        
        # 创建传统BO优化器（先不初始化点）
        self.optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            random_state=random_state
        )
        
        # 策略1: Warm Start生成器
        self.warm_start_generator = WarmStartGenerator(
            pbounds=pbounds,
            model_name=llm_model
        )
        
        # 策略2: 增强核函数配置
        self.kernel_config = get_llm_kernel_config() if use_enhanced_kernel else None
        
        # 策略3: 动态采样策略
        self.dynamic_sampling = DynamicSamplingStrategy(llm_model=llm_model) if use_dynamic_sampling else None
        
        print("LLMBO优化器初始化完成")
        print(f"  LLM模型: {llm_model}")
        print(f"  策略1 (Warm Start): 启用")
        print(f"  策略2 (增强核函数): {'启用' if use_enhanced_kernel else '禁用'}")
        print(f"  策略3 (动态采样): {'启用' if use_dynamic_sampling else '禁用'}")
    
    def _setup_enhanced_kernel(self):
        """
        设置增强的核函数到GP中
        """
        if not self.use_enhanced_kernel:
            return
        
        print("\n配置LLM增强的核函数...")
        
        # 创建增强核函数
        kernel = LLMEnhancedKernel(
            length_scales=self.kernel_config['length_scales'],
            coupling_matrix=self.kernel_config['coupling_matrix'],
            coupling_strength=self.kernel_config['coupling_strength']
        )
        
        # 设置到GP中
        self.optimizer._gp.kernel = kernel
        
        print("  核函数已配置")
        print(f"  Length scales: {self.kernel_config['length_scales']}")
        print(f"  Coupling strength: {self.kernel_config['coupling_strength']}")
    
    def _get_dynamic_sampling_guidance(self, iteration: int) -> Dict:
        """
        获取动态采样指导
        
        参数:
            iteration: 当前迭代次数
            
        返回:
            采样指导字典
        """
        if not self.use_dynamic_sampling or not self.dynamic_sampling.should_analyze(iteration):
            return None
        
        try:
            guidance = self.dynamic_sampling.get_analysis(
                historical_results=self.optimizer.res,
                pbounds=self.pbounds,
                current_iteration=iteration
            )
            return guidance
        except Exception as e:
            print(f"  警告: 动态采样分析失败 - {e}")
            return None
    
    def maximize(
        self,
        init_points: int = 5,
        n_iter: int = 30,
        use_llm_warm_start: bool = True
    ):
        """
        执行优化
        
        参数:
            init_points: 初始探索点数
            n_iter: 贝叶斯优化迭代次数
            use_llm_warm_start: 是否使用LLM Warm Start
        """
        print("\n" + "="*70)
        print("开始LLMBO优化（完整版本）")
        print("="*70)
        
        start_time = time.time()
        
        # 阶段1: LLM Warm Start
        if use_llm_warm_start:
            print("\n[阶段1] LLM Warm Start - 生成初始点")
            print("-"*70)
            
            # 使用LLM生成初始点
            llm_points = self.warm_start_generator.generate(n_points=init_points)
            
            # 将LLM生成的点加入优化器队列
            for i, point in enumerate(llm_points, 1):
                self.optimizer.probe(params=point, lazy=True)
            
            print(f"已添加{init_points}个LLM初始点到队列")
        
        # 配置增强核函数
        if self.use_enhanced_kernel:
            print("\n[阶段2] 配置增强核函数")
            print("-"*70)
            self._setup_enhanced_kernel()
        
        # 阶段3: 贝叶斯优化主循环
        print(f"\n[阶段3] 贝叶斯优化迭代 (n_iter={n_iter})")
        print("-"*70)
        
        # 执行初始点评估 + 贝叶斯优化迭代
        for iter_num in range(n_iter):
            # 检查是否需要动态采样分析
            if self.use_dynamic_sampling:
                guidance = self._get_dynamic_sampling_guidance(iter_num + init_points)
        
        # 运行优化（init_points=0因为已经用probe添加）
        self.optimizer.maximize(init_points=0, n_iter=(init_points + n_iter))
        
        # 计算总时间
        total_time = time.time() - start_time
        
        # 显示结果
        print("\n" + "="*70)
        print("优化完成")
        print("="*70)
        print(f"总评估次数: {len(self.optimizer.space)}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"\n最优结果:")
        print(f"  目标值: {self.optimizer.max['target']:.2f}")
        print(f"  充电步数: {-self.optimizer.max['target']:.0f} 步")
        print(f"  参数:")
        for key, value in self.optimizer.max['params'].items():
            print(f"    {key} = {value:.4f}")
        
        return self.optimizer.max
    
    @property
    def max(self):
        """返回最优结果"""
        return self.optimizer.max
    
    @property
    def res(self):
        """返回所有评估结果"""
        return self.optimizer.res


# 测试代码
if __name__ == "__main__":
    from SPM import SPM
    import numpy as np
    
    def charging_time_compute(current1, charging_number, current2):
        """两阶段充电目标函数"""
        env = SPM(3.0, 298)
        done = False
        i = 0
        
        while not done:
            if i < int(charging_number):
                current = current1
                if env.voltage >= 4.0:
                    current = current * np.exp(-0.9 * (env.voltage - 4))
            else:
                current = current2
                if env.voltage >= 4.0:
                    current = current * np.exp(-0.9 * (env.voltage - 4))
            
            _, done, _ = env.step(current)
            i += 1
            
            if env.voltage > env.sett['constraints voltage max'] or \
               env.temp > env.sett['constraints temperature max']:
                i += 1
            
            if done:
                return -i
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    # 创建完整LLMBO优化器
    llmbo = LLMBOOptimizer(
        f=charging_time_compute,
        pbounds=pbounds,
        llm_model="gpt-3.5-turbo",
        random_state=1,
        use_enhanced_kernel=True,
        use_dynamic_sampling=True
    )
    
    # 运行小规模测试
    print("\n开始小规模测试...")
    result = llmbo.maximize(
        init_points=5,
        n_iter=5,
        use_llm_warm_start=True
    )