"""
测试手动迭代控制
验证：能够手动控制每次BO迭代
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization, UtilityFunction
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


def test_manual_iteration():
    """
    测试手动迭代控制
    """
    print("="*70)
    print("测试手动迭代控制")
    print("="*70)
    
    # 参数边界
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    # 创建BO优化器（不指定目标函数）
    optimizer = BayesianOptimization(
        f=None,  # 不指定目标函数，手动控制
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )
    
    # 创建utility function（UCB）
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    
    print("\n阶段1: 随机初始化（5个点）")
    print("-"*70)
    
    # 手动添加5个随机初始点
    for i in range(5):
        # 建议下一个点（随机）
        next_point = optimizer.suggest(utility)
        
        print(f"\n迭代 {i+1}:")
        print(f"  建议参数: current1={next_point['current1']:.2f}, "
              f"charging_number={next_point['charging_number']:.1f}, "
              f"current2={next_point['current2']:.2f}")
        
        # 评估目标函数
        target = charging_time_compute(**next_point)
        print(f"  评估结果: {-target:.0f} 步")
        
        # 注册结果到优化器
        optimizer.register(params=next_point, target=target)
    
    print("\n" + "="*70)
    print("阶段2: BO迭代（3个点）")
    print("-"*70)
    
    # 手动进行3次BO迭代
    for i in range(3):
        # 建议下一个点（基于GP）
        next_point = optimizer.suggest(utility)
        
        print(f"\n迭代 {5+i+1}:")
        print(f"  建议参数: current1={next_point['current1']:.2f}, "
              f"charging_number={next_point['charging_number']:.1f}, "
              f"current2={next_point['current2']:.2f}")
        
        # 评估目标函数
        target = charging_time_compute(**next_point)
        print(f"  评估结果: {-target:.0f} 步")
        
        # 注册结果
        optimizer.register(params=next_point, target=target)
        
        # 显示当前最优
        current_best = optimizer.max
        print(f"  当前最优: {-current_best['target']:.0f} 步")
    
    # 显示最终结果
    print("\n" + "="*70)
    print("最终结果")
    print("="*70)
    print(f"\n总评估次数: {len(optimizer.space)}")
    print(f"最优充电步数: {-optimizer.max['target']:.0f} 步")
    print(f"最优参数:")
    for key, value in optimizer.max['params'].items():
        print(f"  {key} = {value:.4f}")
    
    # 验证：检查GP是否已训练
    print(f"\nGP训练数据点数: {len(optimizer._gp.X_train_)}")
    print(f"期望: 8 (5个初始点 + 3次迭代)")
    
    if len(optimizer._gp.X_train_) == 8:
        print("\n验证成功: 手动迭代控制正常工作")
        return True
    else:
        print("\n验证失败: GP数据点数量不匹配")
        return False


if __name__ == "__main__":
    success = test_manual_iteration()
    
    if success:
        print("\n" + "="*70)
        print("步骤1完成：手动迭代控制测试通过")
        print("="*70)
        print("\n可以进入步骤2：实现自定义Acquisition Function")
    else:
        print("\n" + "="*70)
        print("步骤1失败：请检查问题")
        print("="*70)