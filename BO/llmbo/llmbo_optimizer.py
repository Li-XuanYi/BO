"""
LLMBO优化器 (核心修复版)
修复内容:
1. 修正maximize的n_iter参数错误
2. 修正gamma更新时机
3. 优化队列处理逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization, Events
from llm_utils import WarmStartGenerator
try:
    from .enhanced_kernel import LLMEnhancedKernel, get_llm_kernel_config
    from .dynamic_sampling import DynamicSamplingStrategy
except ImportError:
    from enhanced_kernel import LLMEnhancedKernel, get_llm_kernel_config
    from dynamic_sampling import DynamicSamplingStrategy
from typing import Callable, Dict
import time


class GammaUpdater:
    """Gamma更新器"""
    def __init__(self, optimizer_instance):
        self.optimizer_instance = optimizer_instance
    
    def update(self, event, instance):
        self.optimizer_instance._update_coupling_strength_internal(event, instance)


class LLMBOOptimizer:
    """LLM增强的贝叶斯优化器 (修复版)"""
    
    def __init__(
        self,
        f: Callable,
        pbounds: Dict[str, tuple],
        llm_model: str = "gpt-3.5-turbo",
        random_state: int = 1,
        use_enhanced_kernel: bool = True,
        use_dynamic_sampling: bool = True
    ):
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
        
        # 策略2: 增强核函数配置
        self.kernel_config = get_llm_kernel_config() if use_enhanced_kernel else None
        self.enhanced_kernel = None
        
        # 策略3: 动态采样策略
        self.dynamic_sampling = DynamicSamplingStrategy(llm_model=llm_model) if use_dynamic_sampling else None
        
        # Gamma更新器
        self.gamma_updater = GammaUpdater(self)
        
        print("LLMBO优化器初始化完成")
        print(f"  LLM模型: {llm_model}")
        print(f"  策略1 (Warm Start): 启用")
        print(f"  策略2 (增强核函数): {'启用' if use_enhanced_kernel else '禁用'}")
        print(f"  策略3 (动态采样): {'启用' if use_dynamic_sampling else '禁用'}")
    
    def _setup_enhanced_kernel(self):
        """设置增强的核函数"""
        if not self.use_enhanced_kernel:
            return
        
        print("\n配置LLM增强的核函数...")
        
        self.enhanced_kernel = LLMEnhancedKernel(
            length_scales=self.kernel_config['length_scales'],
            coupling_matrix=self.kernel_config['coupling_matrix'],
            coupling_strength=self.kernel_config['coupling_strength']
        )
        
        self.optimizer._gp.kernel = self.enhanced_kernel
        
        print("  核函数已配置")
        print(f"  Length scales: {self.kernel_config['length_scales']}")
        print(f"  Coupling strength: {self.kernel_config['coupling_strength']}")
    
    def _update_coupling_strength_internal(self, event, instance):
        """更新核函数的耦合强度gamma (传递优化历史给LLM)"""
        if not self.use_enhanced_kernel or self.enhanced_kernel is None:
            return
        
        # 修复: 只在有至少2个结果时才更新
        if len(instance.res) < 2:
            return
        
        current_best = instance.max['target']
        iteration = len(instance.res) - 1
        
        # 每5次打印一次
        verbose = (iteration % 5 == 0)
        
        # 关键修改: 传递完整优化历史供LLM分析
        self.enhanced_kernel.update_gamma(
            current_f_min=current_best,
            iteration=iteration,
            optimization_history=instance.res,  # 传递历史数据
            verbose=verbose
        )
    
    def _get_dynamic_sampling_guidance(self, iteration: int) -> Dict:
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
        执行优化 (修复版)
        
        修复内容:
        1. 正确的maximize调用: n_iter=n_iter (而非 n_iter=(init_points+n_iter))
        2. 先评估LLM点,再订阅gamma更新事件
        3. 动态采样分析逻辑优化
        """
        print("\n" + "="*70)
        print("开始LLMBO优化 (修复版)")
        print("="*70)
        
        start_time = time.time()
        
        # 阶段1: LLM Warm Start
        if use_llm_warm_start:
            print("\n[阶段1] LLM Warm Start - 生成初始点")
            print("-"*70)
            
            llm_points = self.warm_start_generator.generate(n_points=init_points)
            
            # 将LLM生成的点加入优化器队列
            for i, point in enumerate(llm_points, 1):
                self.optimizer.probe(params=point, lazy=True)
            
            print(f"已添加{init_points}个LLM初始点到队列")
        
        # 阶段2: 配置增强核函数
        if self.use_enhanced_kernel:
            print("\n[阶段2] 配置增强核函数")
            print("-"*70)
            self._setup_enhanced_kernel()
            print("  增强核函数已配置")
        
        # 修复: 先评估LLM初始点
        print(f"\n[阶段2.5] 评估LLM初始点")
        print("-"*70)
        # 执行队列中的点(不做额外迭代)
        self.optimizer.maximize(init_points=0, n_iter=0)
        print(f"  已评估 {len(self.optimizer.res)} 个LLM初始点")
        
        # 修复: 现在才订阅gamma更新事件
        if self.use_enhanced_kernel:
            self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.gamma_updater)
            print("  已启用动态gamma调整")
        
        # 阶段3: 贝叶斯优化主循环
        print(f"\n[阶段3] 贝叶斯优化迭代 (n_iter={n_iter})")
        print("-"*70)
        
        # 动态采样分析(每5次迭代)
        for iter_num in range(n_iter):
            if self.use_dynamic_sampling:
                current_iter = len(self.optimizer.res) + iter_num
                guidance = self._get_dynamic_sampling_guidance(current_iter)
        
        # 修复: 正确的n_iter参数
        self.optimizer.maximize(init_points=0, n_iter=n_iter)
        
        # 计算总时间
        total_time = time.time() - start_time
        
        # 显示结果
        print("\n" + "="*70)
        print("优化完成")
        print("="*70)
        print(f"总评估次数: {len(self.optimizer.space)}")
        print(f"预期评估次数: {init_points + n_iter}")
        print(f"总耗时: {total_time:.2f} 秒")
        
        # 显示gamma变化历史
        if self.enhanced_kernel:
            history = self.enhanced_kernel.get_gamma_history()
            print(f"\n策略2: gamma动态调整统计")
            print("-"*70)
            print(f"  初始gamma: {history['gamma_history'][0]:.4f}")
            print(f"  最终gamma: {history['gamma_history'][-1]:.4f}")
            print(f"  变化幅度: {(history['gamma_history'][-1] - history['gamma_history'][0]):.4f}")
            if len(history['gamma_history']) > 10:
                print(f"  gamma轨迹: [{', '.join([f'{g:.3f}' for g in history['gamma_history'][:3]])}] ... "
                      f"[{', '.join([f'{g:.3f}' for g in history['gamma_history'][-3:]])}]")
            else:
                print(f"  gamma轨迹: [{', '.join([f'{g:.3f}' for g in history['gamma_history']])}]")
        
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