"""
LLMBO优化器 V3 - 聚焦核心创新

整合三个LLM策略:
1. ✅ LLM Warm Start (方程4) - 智能初始化
2. ✅ LLM增强核函数 (方程5-7) - 启发式梯度 + 智能gamma调整  
3. ⭐ LLM增强采样 (方程8-9) - 核心创新！智能候选点选择

关键改进:
- 避免实时SPM计算，保持效率
- 聚焦LLM智能决策
- 三个策略协同工作

论文参考: manuscript1.pdf
"""

import sys
import os
from bayes_opt import BayesianOptimization, Events
from bayes_opt import UtilityFunction
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from typing import Callable, Dict, Optional, List
import numpy as np


class LLMBOOptimizerV3:
    """
    LLMBO优化器V3 - 聚焦论文核心创新
    
    三大策略:
    1. LLM Warm Start: 智能初始点生成
    2. LLM Enhanced Kernel: 增强核函数（启发式梯度）
    3. LLM Enhanced Sampling: 智能候选采样（核心！）
    """
    
    def __init__(
        self,
        f: Callable,
        pbounds: Dict[str, tuple],
        llm_model: str = "gpt-3.5-turbo",
        random_state: Optional[int] = None,
        use_warm_start: bool = True,
        use_enhanced_kernel: bool = True,
        use_enhanced_sampling: bool = True,
    ):
        """
        初始化LLMBO优化器V3
        
        参数:
            f: 目标函数
            pbounds: 参数边界字典
            llm_model: LLM模型名称
            random_state: 随机种子（None表示真随机）
            use_warm_start: 是否使用LLM Warm Start
            use_enhanced_kernel: 是否使用增强核函数
            use_enhanced_sampling: 是否使用LLM增强采样
        """
        self.f = f
        self.pbounds = pbounds
        self.llm_model = llm_model
        self.random_state = random_state
        self.use_warm_start = use_warm_start
        self.use_enhanced_kernel = use_enhanced_kernel
        self.use_enhanced_sampling = use_enhanced_sampling
        
        # 参数名称
        self.param_names = tuple(pbounds.keys())
        
        # 创建传统BO优化器（作为基础）
        self.optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            random_state=random_state
        )
        
        # 策略1: Warm Start生成器
        self.warm_start_generator = None
        if use_warm_start:
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from llm_utils import WarmStartGenerator
                self.warm_start_generator = WarmStartGenerator(
                    pbounds=pbounds,
                    model_name=llm_model
                )
                print("[策略1] LLM Warm Start: ✅ 启用")
            except Exception as e:
                print(f"[策略1] LLM Warm Start: ⚠️ 初始化失败 - {e}")
                self.use_warm_start = False
        
        # 策略2: 增强核函数
        self.enhanced_kernel = None
        if use_enhanced_kernel:
            try:
                # 注意：这里需要导入我们刚创建的V3版本
                # 实际使用时需要将enhanced_kernel_v3.py放到正确位置
                from enhanced_kernel_v3 import LLMEnhancedKernelV3, get_llm_kernel_config_v3
                
                kernel_config = get_llm_kernel_config_v3(param_names=self.param_names)
                self.enhanced_kernel = LLMEnhancedKernelV3(
                    param_names=kernel_config['param_names'],
                    length_scales=kernel_config['length_scales'],
                    coupling_matrix=kernel_config['coupling_matrix'],
                    coupling_strength=kernel_config['coupling_strength'],
                    use_llm_guidance=True,
                    llm_model=llm_model
                )
                print("[策略2] LLM Enhanced Kernel: ✅ 启用")
            except Exception as e:
                print(f"[策略2] LLM Enhanced Kernel: ⚠️ 初始化失败 - {e}")
                self.use_enhanced_kernel = False
        
        # 策略3: LLM增强采样（核心！）
        self.enhanced_sampler = None
        if use_enhanced_sampling:
            try:
                from dynamic_sampling_v2 import LLMEnhancedSamplingV2
                self.enhanced_sampler = LLMEnhancedSamplingV2(
                    llm_model=llm_model,
                    analysis_interval=3,
                    min_data_points=3
                )
                print("[策略3] LLM Enhanced Sampling: ⭐ 启用（核心创新）")
            except Exception as e:
                print(f"[策略3] LLM Enhanced Sampling: ⚠️ 初始化失败 - {e}")
                self.use_enhanced_sampling = False
        
        # 优化历史
        self.iteration_count = 0
        self.f_min_history = []
        
        # 注册事件回调
        if self.use_enhanced_kernel:
            self.optimizer.subscribe(
                Events.OPTIMIZATION_STEP,
                self._on_step
            )
        
        print("\n[LLMBO-V3] 初始化完成")
        print(f"  LLM模型: {llm_model}")
        print(f"  随机种子: {random_state}")
        print(f"  参数: {self.param_names}")
    
    def _on_step(self, event, instance):
        """每步优化后的回调 - 更新gamma"""
        if not self.use_enhanced_kernel or self.enhanced_kernel is None:
            return
        
        # 获取当前最优值
        if len(self.optimizer.space) == 0:
            return
        
        current_best = self.optimizer.max['target']
        self.f_min_history.append(current_best)
        
        # 更新gamma（每步都更新，但LLM只在特定迭代参与）
        if len(self.f_min_history) >= 2:
            f_min_prev = self.f_min_history[-2]
            f_min_curr = self.f_min_history[-1]
            
            # 获取历史结果
            historical_results = []
            for i in range(len(self.optimizer.space)):
                point = self.optimizer.space.params[i]
                target = self.optimizer.space.target[i]
                historical_results.append({
                    'params': point,
                    'target': target
                })
            
            # 智能更新gamma
            self.enhanced_kernel.smart_update_gamma(
                iteration=self.iteration_count,
                f_min_prev=f_min_prev,
                f_min_curr=f_min_curr,
                historical_results=historical_results,
                verbose=(self.iteration_count % 3 == 0)  # 每3次迭代打印一次
            )
        
        self.iteration_count += 1
    
    def _generate_warm_start_points(self, n_points: int) -> List[Dict]:
        """
        生成LLM Warm Start初始点（策略1）
        
        参数:
            n_points: 需要生成的点数
        
        返回:
            初始点列表
        """
        if not self.use_warm_start or self.warm_start_generator is None:
            print(f"  使用随机初始化（{n_points}个点）")
            return []
        
        try:
            print(f"  使用LLM Warm Start生成{n_points}个智能初始点...")
            points = self.warm_start_generator.generate_warm_start_points(n_points=n_points)
            print(f"  ✅ 成功生成{len(points)}个LLM初始点")
            return points
        except Exception as e:
            print(f"  ⚠️ LLM Warm Start失败: {e}")
            print(f"  回退到随机初始化")
            return []
    
    def _setup_enhanced_gp(self):
        """设置使用增强核函数的GP"""
        if not self.use_enhanced_kernel or self.enhanced_kernel is None:
            return
        
        try:
            # 创建使用增强核的GP
            gp = GaussianProcessRegressor(
                kernel=self.enhanced_kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.random_state
            )
            
            # 替换optimizer的GP
            self.optimizer._gp = gp
            print("  ✅ 增强核函数已应用到GP")
        except Exception as e:
            print(f"  ⚠️ 设置增强核失败: {e}")
    
    def _get_next_point_with_llm_sampling(self) -> Dict:
        """
        使用LLM增强采样获取下一个候选点（策略3 - 核心！）
        
        这是论文的核心创新！
        """
        if not self.use_enhanced_sampling or self.enhanced_sampler is None:
            # 回退到标准采样
            utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
            return self.optimizer.suggest(utility)
        
        try:
            # 获取历史结果
            historical_results = []
            for i in range(len(self.optimizer.space)):
                point = self.optimizer.space.params[i]
                target = self.optimizer.space.target[i]
                historical_results.append({
                    'params': point,
                    'target': target
                })
            
            # LLM分析并更新权重
            self.enhanced_sampler.analyze_and_update_weights(
                historical_results=historical_results,
                pbounds=self.pbounds,
                current_iteration=self.iteration_count,
                verbose=(self.iteration_count % 3 == 0)
            )
            
            # 获取采样区域建议
            region_advice = self.enhanced_sampler.suggest_next_sample_region(
                historical_results=historical_results,
                pbounds=self.pbounds
            )
            
            # 根据策略调整效用函数
            if region_advice['strategy'] == 'EXPLOITATION':
                # 开发模式：降低kappa，更倾向于开发
                utility = UtilityFunction(kind="ucb", kappa=1.5, xi=0.0)
            elif region_advice['strategy'] == 'EXPLORATION':
                # 探索模式：提高kappa，更倾向于探索
                utility = UtilityFunction(kind="ucb", kappa=3.5, xi=0.0)
            else:  # BALANCED
                # 平衡模式：标准kappa
                utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
            
            # 获取建议点
            suggestion = self.optimizer.suggest(utility)
            
            # 应用LLM权重调整（如果在聚焦区域内）
            # 这里可以进一步优化，确保采样点在LLM建议的区域内
            # 但为了保持简单，我们先使用标准suggest
            
            return suggestion
            
        except Exception as e:
            print(f"  ⚠️ LLM采样失败: {e}, 使用标准采样")
            utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
            return self.optimizer.suggest(utility)
    
    def maximize(
        self,
        init_points: int = 5,
        n_iter: int = 25,
        verbose: int = 2
    ) -> Dict:
        """
        运行LLMBO优化
        
        参数:
            init_points: 初始点数量（使用LLM Warm Start）
            n_iter: 贝叶斯优化迭代次数
            verbose: 日志详细程度 (0=静默, 1=简要, 2=详细)
        
        返回:
            最优结果字典 {'target': ..., 'params': {...}}
        """
        print("\n" + "="*70)
        print("LLMBO优化开始")
        print("="*70)
        
        # 阶段1: LLM Warm Start初始化
        print(f"\n[阶段1] 初始化 ({init_points}个点)")
        print("-"*70)
        
        warm_start_points = self._generate_warm_start_points(init_points)
        
        if len(warm_start_points) > 0:
            # 使用LLM生成的点
            for i, point in enumerate(warm_start_points):
                if verbose >= 2:
                    print(f"  LLM点{i+1}: {point}")
                target = self.f(**point)
                self.optimizer.register(params=point, target=target)
                if verbose >= 1:
                    print(f"    → 充电步数: {-target:.0f} 步")
        else:
            # 回退到随机初始化
            for i in range(init_points):
                random_point = {}
                for param, (low, high) in self.pbounds.items():
                    random_point[param] = np.random.uniform(low, high)
                if verbose >= 2:
                    print(f"  随机点{i+1}: {random_point}")
                target = self.f(**random_point)
                self.optimizer.register(params=random_point, target=target)
                if verbose >= 1:
                    print(f"    → 充电步数: {-target:.0f} 步")
        
        # 设置增强核函数
        if self.use_enhanced_kernel:
            self._setup_enhanced_gp()
        
        # 阶段2: 贝叶斯优化迭代
        print(f"\n[阶段2] 贝叶斯优化 ({n_iter}次迭代)")
        print("-"*70)
        
        for i in range(n_iter):
            if verbose >= 1:
                print(f"\n迭代 {i+1}/{n_iter}")
            
            # 使用LLM增强采样获取下一个点
            next_point = self._get_next_point_with_llm_sampling()
            
            if verbose >= 2:
                print(f"  候选点: {next_point}")
            
            # 评估
            target = self.f(**next_point)
            self.optimizer.register(params=next_point, target=target)
            
            if verbose >= 1:
                current_best = self.optimizer.max['target']
                print(f"  充电步数: {-target:.0f} 步")
                print(f"  当前最优: {-current_best:.0f} 步")
        
        # 返回最优结果
        print("\n" + "="*70)
        print("优化完成")
        print("="*70)
        
        return self.optimizer.max
    
    def get_optimization_history(self) -> Dict:
        """获取优化历史"""
        history = {
            'evaluations': [],
            'best_per_iteration': []
        }
        
        # 所有评估点
        for i in range(len(self.optimizer.space)):
            point = self.optimizer.space.params[i]
            target = self.optimizer.space.target[i]
            history['evaluations'].append({
                'iteration': i,
                'params': point,
                'target': target,
                'charging_steps': -target
            })
        
        # 每次迭代的最优值
        current_best = float('-inf')
        for eval_data in history['evaluations']:
            if eval_data['target'] > current_best:
                current_best = eval_data['target']
            history['best_per_iteration'].append({
                'iteration': eval_data['iteration'],
                'best_target': current_best,
                'best_charging_steps': -current_best
            })
        
        # Gamma历史（如果使用增强核）
        if self.use_enhanced_kernel and self.enhanced_kernel:
            gamma_hist = self.enhanced_kernel.get_gamma_history()
            history['gamma_history'] = gamma_hist['gamma_history']
            history['f_min_history'] = gamma_hist['f_min_history']
        
        # 采样策略历史（如果使用增强采样）
        if self.use_enhanced_sampling and self.enhanced_sampler:
            history['sampling_stats'] = self.enhanced_sampler.get_statistics()
        
        return history


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("LLMBO优化器V3 - 单元测试")
    print("="*70)
    
    # 简单测试函数（模拟充电优化）
    def test_function(current1, charging_number, current2):
        """模拟充电函数"""
        optimal_c1 = 4.5
        optimal_cn = 12.0
        optimal_c2 = 2.5
        
        # 简单的二次函数模拟
        error = (
            (current1 - optimal_c1)**2 + 
            (charging_number - optimal_cn)**2 / 10 +
            (current2 - optimal_c2)**2
        )
        
        # 返回负值（最小化充电步数）
        return -(80 + error * 10)
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    print("\n创建LLMBO优化器...")
    llmbo = LLMBOOptimizerV3(
        f=test_function,
        pbounds=pbounds,
        llm_model="gpt-3.5-turbo",
        random_state=42,
        use_warm_start=False,  # 测试时先禁用
        use_enhanced_kernel=False,  # 测试时先禁用
        use_enhanced_sampling=False  # 测试时先禁用
    )
    
    print("\n运行优化...")
    result = llmbo.maximize(
        init_points=3,
        n_iter=5,
        verbose=2
    )
    
    print("\n最优结果:")
    print(f"  充电步数: {-result['target']:.0f}")
    print(f"  参数: {result['params']}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)
