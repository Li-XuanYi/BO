"""
LLMBO优化器 V2 - 集成梯度估计和改进的增强核函数

主要改进:
1. 使用 enhanced_kernel_v2 (基于梯度的耦合项)
2. 自动创建和管理 GradientEstimator
3. 完全兼容原有API
"""

import sys
import os
from bayes_opt import BayesianOptimization, Events
from typing import Callable, Dict, Optional
import time
import numpy as np

# 导入新模块
try:
    from llmbo.enhanced_kernel import LLMEnhancedKernel, get_llm_kernel_config
    from llmbo.dynamic_sampling import DynamicSamplingStrategy
    from llm_utils import WarmStartGenerator
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llmbo.enhanced_kernel import LLMEnhancedKernel, get_llm_kernel_config
    from llmbo.dynamic_sampling import DynamicSamplingStrategy
    from llm_utils import WarmStartGenerator


class GammaUpdater:
    """Gamma更新器"""
    def __init__(self, optimizer_instance):
        self.optimizer_instance = optimizer_instance
    
    def update(self, event, instance):
        self.optimizer_instance._update_coupling_strength_internal(event, instance)


class LLMBOOptimizerV2:
    """
    LLMBO优化器 V2 - 集成梯度估计
    
    使用方法与原版相同,但自动使用基于梯度的增强核函数
    """
    
    def __init__(
        self,
        f: Callable,
        pbounds: Dict[str, tuple],
        llm_model: str = "gpt-3.5-turbo",
        random_state: int = 1,
        use_enhanced_kernel: bool = True,
        use_dynamic_sampling: bool = True,
        gradient_epsilon: float = 1.0,
        gradient_cache_size: int = 1000
    ):
        """
        初始化LLMBO优化器V2
        
        参数:
            f: 目标函数
            pbounds: 参数边界字典
            llm_model: LLM模型名称
            random_state: 随机种子
            use_enhanced_kernel: 是否使用增强核函数
            use_dynamic_sampling: 是否使用动态采样
            gradient_epsilon: 梯度估计的差分步长
            gradient_cache_size: 梯度缓存大小
        """
        self.f = f
        self.pbounds = pbounds
        self.llm_model = llm_model
        self.random_state = random_state
        self.use_enhanced_kernel = use_enhanced_kernel
        self.use_dynamic_sampling = use_dynamic_sampling
        
        # 创建传统BO优化器
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
        
        # # 新增: 梯度估计器
        # if use_enhanced_kernel:
        #     param_names = tuple(pbounds.keys())
        #     print(f"\n[LLMBO-V2] 初始化梯度估计器")
        #     print(f"  参数: {param_names}")
        #     print(f"  Epsilon: {gradient_epsilon}")
            

        # else:
        #     self.gradient_estimator = None
        
        # 策略2: 增强核函数配置 (使用V2)
        self.kernel_config = get_llm_kernel_config() if use_enhanced_kernel else None
        self.enhanced_kernel = None
        
        # 策略3: 动态采样策略
        self.dynamic_sampling = DynamicSamplingStrategy(llm_model=llm_model) if use_dynamic_sampling else None
        
        # Gamma更新器
        self.gamma_updater = GammaUpdater(self)
        
        print("\n[LLMBO-V2] 优化器初始化完成")
        print(f"  LLM模型: {llm_model}")
        print(f"  策略1 (Warm Start): 启用")
        print(f"  策略2 (增强核函数V2): {'启用' if use_enhanced_kernel else '禁用'}")
        print(f"  策略3 (动态采样): {'启用' if use_dynamic_sampling else '禁用'}")
        print(f"  梯度估计: {'启用' if use_enhanced_kernel else '禁用'}")
    
    def _setup_enhanced_kernel(self):
        """设置增强的核函数 (V2版本)"""
        if not self.use_enhanced_kernel:
            return
        
        print("\n[LLMBO-V2] 配置增强核函数V2...")
        
        self.enhanced_kernel = LLMEnhancedKernel(
            length_scales=self.kernel_config['length_scales'],
            coupling_matrix=self.kernel_config['coupling_matrix'],
            coupling_strength=self.kernel_config['coupling_strength'],
            use_llm_guidance=True,
            llm_model=self.llm_model
        )
        
        self.optimizer._gp.kernel = self.enhanced_kernel
        
        print("  核函数已配置 (基于梯度)")
        print(f"  Length scales: {self.kernel_config['length_scales']}")
        print(f"  Coupling strength: {self.kernel_config['coupling_strength']}")
    
    def _update_coupling_strength_internal(self, event, instance):
        """更新核函数的耦合强度gamma"""
        if not self.use_enhanced_kernel or self.enhanced_kernel is None:
            return
        
        if len(instance.res) < 2:
            return
        
        current_best = instance.max['target']
        iteration = len(instance.res) - 1
        
        verbose = (iteration % 5 == 0)
        
        self.enhanced_kernel.update_gamma(
            current_f_min=current_best,
            iteration=iteration,
            optimization_history=instance.res,
            verbose=verbose
        )
    
    def _get_dynamic_sampling_guidance(self, iteration: int) -> Optional[Dict]:
        """获取动态采样指导"""
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
            init_points: LLM生成的初始点数量
            n_iter: 贝叶斯优化迭代次数
            use_llm_warm_start: 是否使用LLM Warm Start
        """
        print("\n" + "="*70)
        print("开始LLMBO-V2优化 (梯度增强版)")
        print("="*70)
        
        start_time = time.time()
        
        # 阶段1: LLM Warm Start
        if use_llm_warm_start:
            print("\n[阶段1] LLM Warm Start - 生成初始点")
            print("-"*70)
            
            llm_points = self.warm_start_generator.generate(n_points=init_points)
            
            for i, point in enumerate(llm_points, 1):
                self.optimizer.probe(params=point, lazy=True)
            
            print(f"已添加{init_points}个LLM初始点到队列")
        
        # 阶段2: 配置增强核函数
        if self.use_enhanced_kernel:
            print("\n[阶段2] 配置增强核函数V2")
            print("-"*70)
            self._setup_enhanced_kernel()
        
        # 阶段2.5: 评估LLM初始点
        print(f"\n[阶段2.5] 评估LLM初始点")
        print("-"*70)
        self.optimizer.maximize(init_points=0, n_iter=0)
        print(f"  已评估 {len(self.optimizer.res)} 个LLM初始点")
        
        # 订阅gamma更新事件
        if self.use_enhanced_kernel:
            self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.gamma_updater)
            print("  已启用动态gamma调整")
        
        # 阶段3: BO主循环
        print(f"\n[阶段3] 贝叶斯优化迭代 (n_iter={n_iter})")
        print("-"*70)
        
        self.optimizer.maximize(init_points=0, n_iter=n_iter)
        
        total_time = time.time() - start_time
        
        # 显示结果
        print("\n" + "="*70)
        print("优化完成")
        print("="*70)
        print(f"总评估次数: {len(self.optimizer.space)}")
        print(f"总耗时: {total_time:.2f} 秒")
        
        # 显示gamma历史
        if self.enhanced_kernel:
            history = self.enhanced_kernel.get_gamma_history()
            print(f"\nGamma演化:")
            print(f"  初始: {history['gamma_history'][0]:.4f}")
            print(f"  最终: {history['gamma_history'][-1]:.4f}")
            
            # 显示梯度缓存统计
            cache_stats = self.enhanced_kernel.get_cache_stats()
            print(f"\n梯度缓存统计:")
            print(f"  缓存大小: {cache_stats['gradient_cache_size']}")
            if 'cache_hits' in cache_stats:
                print(f"  命中率: {cache_stats['hit_rate']:.2%}")
        
        # 返回最优结果
        best = self.optimizer.max
        
        print(f"\n最优结果:")
        print(f"  目标函数值: {best['target']:.6f}")
        print(f"  最优参数:")
        for param, value in best['params'].items():
            print(f"    {param} = {value:.4f}")
        
        return best
    
    def get_optimization_history(self) -> Dict:
        """
        获取完整的优化历史
        
        返回:
            包含所有评估点和gamma历史的字典
        """
        history = {
            'evaluations': self.optimizer.res,
            'best_point': self.optimizer.max
        }
        
        if self.enhanced_kernel:
            history.update(self.enhanced_kernel.get_gamma_history())
        
        return history


def test_llmbo_v2():
    """
    测试LLMBO-V2优化器
    """
    print("="*70)
    print("LLMBO-V2 集成测试")
    print("="*70)
    
    # 定义测试函数
    def test_function(x, y, z):
        """简单的测试函数"""
        return -(x - 2)**2 - (y - 3)**2 - (z - 4)**2
    
    pbounds = {
        'x': (0, 5),
        'y': (0, 5),
        'z': (0, 5)
    }
    
    # 创建优化器
    optimizer = LLMBOOptimizerV2(
        f=test_function,
        pbounds=pbounds,
        llm_model="gpt-3.5-turbo",
        random_state=42,
        use_enhanced_kernel=True,
        gradient_epsilon=0.5
    )
    
    # 运行优化
    result = optimizer.maximize(
        init_points=3,
        n_iter=10,
        use_llm_warm_start=True
    )
    
    print("\n测试完成")
    print(f"理论最优值: (2, 3, 4) -> 0")
    print(f"找到的最优值: ({result['params']['x']:.2f}, "
          f"{result['params']['y']:.2f}, "
          f"{result['params']['z']:.2f}) -> {result['target']:.4f}")


if __name__ == "__main__":
    test_llmbo_v2()