"""
LLMBO优化器 (添加充电过程数据记录功能)

新增功能:
- 记录最优参数的完整充电过程
- 保存时间序列数据用于可视化
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
import numpy as np


def record_charging_process(charging_func_with_logging, current1, charging_number, current2):
    """
    记录完整充电过程的时间序列数据
    
    返回:
        {
            'time': [t1, t2, ...],
            'soc': [soc1, soc2, ...],
            'voltage': [v1, v2, ...],
            'current': [i1, i2, ...],
            'temperature': [T1, T2, ...],
            'total_steps': N
        }
    """
    from SPM import SPM
    
    env = SPM(3.0, 298)
    
    time_steps = []
    soc_history = []
    voltage_history = []
    current_history = []
    temp_history = []
    
    done = False
    i = 0
    t = 0
    
    while not done:
        # 确定当前电流
        if i < int(charging_number):
            current = current1
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        else:
            current = current2
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        
        # 记录数据
        time_steps.append(t)
        soc_history.append(env.soc * 100)  # 转换为百分比
        voltage_history.append(env.voltage)
        current_history.append(current)
        temp_history.append(env.temp - 273.15)  # 转换为摄氏度
        
        # 执行一步
        _, done, _ = env.step(current)
        i += 1
        t += env.sett['sample_time'] / 60  # 转换为分钟
        
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 1
        
        if done:
            break
    
    return {
        'time': np.array(time_steps),
        'soc': np.array(soc_history),
        'voltage': np.array(voltage_history),
        'current': np.array(current_history),
        'temperature': np.array(temp_history),
        'total_steps': i,
        'total_time': t
    }


class GammaUpdater:
    """Gamma更新器"""
    def __init__(self, optimizer_instance):
        self.optimizer_instance = optimizer_instance
    
    def update(self, event, instance):
        self.optimizer_instance._update_coupling_strength_internal(event, instance)


class LLMBOOptimizer:
    """LLM增强的贝叶斯优化器 (带数据记录)"""
    
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
        use_llm_warm_start: bool = True,
        record_best: bool = True  # 新增: 是否记录最优结果
    ):
        """
        执行优化 (修复版 + 数据记录)
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
            
            for i, point in enumerate(llm_points, 1):
                self.optimizer.probe(params=point, lazy=True)
            
            print(f"已添加{init_points}个LLM初始点到队列")
        
        # 阶段2: 配置增强核函数
        if self.use_enhanced_kernel:
            print("\n[阶段2] 配置增强核函数")
            print("-"*70)
            self._setup_enhanced_kernel()
            print("  增强核函数已配置")
        
        # 阶段2.5: 先评估LLM初始点
        print(f"\n[阶段2.5] 评估LLM初始点")
        print("-"*70)
        self.optimizer.maximize(init_points=0, n_iter=0)
        print(f"  已评估 {len(self.optimizer.res)} 个LLM初始点")
        
        # 现在订阅gamma更新事件
        if self.use_enhanced_kernel:
            self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.gamma_updater)
            print("  已启用动态gamma调整")
        
        # 阶段3: BO主循环
        print(f"\n[阶段3] 贝叶斯优化迭代 (n_iter={n_iter})")
        print("-"*70)
        
        for iter_num in range(n_iter):
            if self.use_dynamic_sampling:
                current_iter = len(self.optimizer.res) + iter_num
                guidance = self._get_dynamic_sampling_guidance(current_iter)
        
        self.optimizer.maximize(init_points=0, n_iter=n_iter)
        
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
        
        # 新增: 记录最优参数的完整充电过程
        best_data = None
        if record_best:
            print(f"\n记录最优参数的充电过程数据...")
            best_params = self.optimizer.max['params']
            best_data = record_charging_process(
                self.f,
                current1=best_params['current1'],
                charging_number=best_params['charging_number'],
                current2=best_params['current2']
            )
            print(f"  已记录 {len(best_data['time'])} 个时间步数据")
        
        result = {
            'optimizer_result': self.optimizer.max,
            'charging_data': best_data,  # 新增: 充电过程数据
            'gamma_history': history if self.enhanced_kernel else None,
            'total_time': total_time,
            'method': 'LLMBO'
        }
        
        return result
    
    @property
    def max(self):
        """返回最优结果"""
        return self.optimizer.max
    
    @property
    def res(self):
        """返回所有评估结果"""
        return self.optimizer.res