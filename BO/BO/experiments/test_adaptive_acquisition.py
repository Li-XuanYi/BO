"""
测试自适应Acquisition Function（保守策略）
目标：验证自定义acquisition function可以正常工作，不做激进调整
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'llmbo'))
from llmbo.adaptive_acquisition import AdaptiveAcquisition
from SPM import SPM
import numpy as np


def charging_time_compute(current1, charging_number, current2):
    """
    两阶段充电目标函数
    """
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


def test_adaptive_acquisition():
    """
    测试自适应acquisition function（保守策略）
    """
    print("="*70)
    print("测试自适应Acquisition Function (保守策略)")
    print("="*70)
    
    # 参数边界
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    # 创建自适应acquisition function
    # 使用保守的衰减策略，不做激进调整
    print("\n创建AdaptiveAcquisition...")
    acquisition_func = AdaptiveAcquisition(
        kappa=2.5,           # 初始探索参数（标准值）
        kappa_decay=0.98,    # 温和衰减（每次迭代衰减2%）
        random_state=42
    )
    
    # 创建BO优化器
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        acquisition_function=acquisition_func,
        random_state=42,
        verbose=0
    )
    
    print(f"策略: 使用温和衰减（0.98），不做规则式调整")
    print(f"理由: 前期探索本来就不稳定，等步骤3集成LLM后再做智能调整")
    
    print("\n" + "="*70)
    print("阶段1: 初始探索（5个点）")
    print("-"*70)
    
    # 初始探索
    for i in range(5):
        next_point = optimizer.suggest()
        target = charging_time_compute(**next_point)
        optimizer.register(params=next_point, target=target)
        
        print(f"迭代 {i+1}: 充电步数={-target:.0f}, kappa={acquisition_func.current_kappa:.3f}")
    
    print("\n" + "="*70)
    print("阶段2: BO迭代（5个点）")
    print("-"*70)
    
    # BO迭代 - 只使用自然衰减，不做人为调整
    for i in range(5):
        next_point = optimizer.suggest()
        target = charging_time_compute(**next_point)
        optimizer.register(params=next_point, target=target)
        
        print(f"迭代 {5+i+1}: 充电步数={-target:.0f}, "
              f"当前最优={-optimizer.max['target']:.0f}, "
              f"kappa={acquisition_func.current_kappa:.3f}")
    
    # 显示结果
    print("\n" + "="*70)
    print("最终结果")
    print("="*70)
    
    print(f"\n总评估次数: {len(optimizer.space)}")
    print(f"最优充电步数: {-optimizer.max['target']:.0f} 步")
    print(f"最优参数:")
    for key, value in optimizer.max['params'].items():
        print(f"  {key} = {value:.4f}")
    
    # 显示acquisition function统计
    stats = acquisition_func.get_stats()
    print(f"\nAcquisition Function统计:")
    print(f"  初始kappa: {stats['initial_kappa']:.3f}")
    print(f"  最终kappa: {stats['current_kappa']:.3f}")
    print(f"  平均kappa: {stats['average_kappa']:.3f}")
    print(f"  衰减率: 每次迭代保留{0.98*100:.0f}%")
    if len(stats['kappa_history']) >= 10:
        print(f"  kappa变化: {' -> '.join([f'{k:.2f}' for k in stats['kappa_history'][:5]])} ... "
              f"{' -> '.join([f'{k:.2f}' for k in stats['kappa_history'][-3:]])}")
    
    # 验证
    print("\n" + "="*70)
    print("验证结果")
    print("="*70)
    
    # 验证迭代次数
    if stats['total_iterations'] == 10 and len(optimizer.space) == 10:
        print("\n验证成功:")
        print("  1. 自定义acquisition function正常工作")
        print("  2. kappa温和衰减（从2.5 -> {:.2f}）".format(stats['current_kappa']))
        print("  3. 迭代控制正常")
        print("  4. 为步骤3（LLM集成）做好准备")
        return True
    else:
        print("\n验证失败:")
        print(f"  期望迭代: 10, 实际: {stats['total_iterations']}")
        print(f"  期望评估: 10, 实际: {len(optimizer.space)}")
        return False


if __name__ == "__main__":
    print("\n注意:")
    print("  - 本测试使用保守的衰减策略（0.98）")
    print("  - 不做规则式的kappa调整")
    print("  - 理由: 论文中的自适应策略是基于LLM分析，而不是简单规则")
    print("  - 步骤3将集成LLM来实现真正的智能调整\n")
    
    success = test_adaptive_acquisition()
    
    if success:
        print("\n" + "="*70)
        print("步骤2完成: 自定义Acquisition Function测试通过")
        print("="*70)
        print("\n下一步: 步骤3 - 集成LLM动态采样")
        print("  - LLM分析历史优化数据")
        print("  - 智能调整采样策略")
        print("  - 参考LLMBO论文的权重函数方法")
    else:
        print("\n" + "="*70)
        print("步骤2失败: 请检查问题")
        print("="*70)