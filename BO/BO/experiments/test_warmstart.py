"""
测试LLM Warm Start是否正确集成到BO中
验证：
1. LLM生成的5个点是否被BO接受
2. 这5个点是否被用于训练GP
3. 第6次迭代是否基于这5个点的GP模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization
from llm_utils import WarmStartGenerator
import numpy as np


def dummy_function(current1, charging_number, current2):
    """
    简单的测试函数（不调用SPM，快速测试）
    """
    # 简单的二次函数，最优点在 (4.5, 15, 2.5)
    return -(
        (current1 - 4.5)**2 + 
        (charging_number - 15)**2 * 0.1 + 
        (current2 - 2.5)**2
    )


def test_warm_start_integration():
    """
    测试Warm Start集成
    """
    print("="*70)
    print("测试LLM Warm Start集成到BO")
    print("="*70)
    
    # 参数边界
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    # 创建BO优化器
    optimizer = BayesianOptimization(
        f=dummy_function,
        pbounds=pbounds,
        random_state=1,
        verbose=0  # 关闭日志，便于观察
    )
    
    print("\n步骤1: 检查初始状态")
    print("-"*70)
    print(f"BO内部数据点数量: {len(optimizer.space)}")
    print(f"期望: 0（还没添加任何点）")
    
    # 生成LLM初始点
    print("\n步骤2: 生成LLM初始点")
    print("-"*70)
    generator = WarmStartGenerator(pbounds=pbounds, model_name="gpt-3.5-turbo")
    llm_points = generator.generate(n_points=5)
    
    # 将LLM点加入BO
    print("\n步骤3: 将LLM点加入BO队列")
    print("-"*70)
    for i, point in enumerate(llm_points, 1):
        print(f"添加点 {i}: current1={point['current1']:.2f}, "
              f"charging_number={point['charging_number']:.2f}, "
              f"current2={point['current2']:.2f}")
        optimizer.probe(params=point, lazy=True)
    
    print(f"\nBO队列长度: {len(optimizer._queue)}")
    print(f"期望: 5（5个点在队列中等待评估）")
    
    # 运行优化（0个随机点 + 2次迭代）
    print("\n步骤4: 运行优化（init_points=0, n_iter=2）")
    print("-"*70)
    optimizer.maximize(init_points=0, n_iter=2, verbose=0)
    
    # 检查结果
    print("\n步骤5: 检查优化后的状态")
    print("-"*70)
    print(f"BO内部数据点数量: {len(optimizer.space)}")
    print(f"期望: 7（5个LLM点 + 2次迭代 = 7次评估）")
    
    # 显示所有评估点
    print("\n步骤6: 显示所有评估点")
    print("-"*70)
    print(f"{'#':<4} {'来源':<15} {'current1':<10} {'charging_number':<18} {'current2':<10} {'目标值':<10}")
    print("-"*70)
    
    for i, res in enumerate(optimizer.res, 1):
        source = "LLM初始点" if i <= 5 else "BO迭代"
        print(f"{i:<4} {source:<15} "
              f"{res['params']['current1']:<10.4f} "
              f"{res['params']['charging_number']:<18.4f} "
              f"{res['params']['current2']:<10.4f} "
              f"{res['target']:<10.4f}")
    
    # 验证GP是否使用了这些点
    print("\n步骤7: 验证GP训练")
    print("-"*70)
    print(f"GP训练数据点数量: {len(optimizer._gp.X_train_)}")
    print(f"期望: 7")
    
    if len(optimizer._gp.X_train_) == 7:
        print("验证成功: GP使用了所有7个点（包括5个LLM点）")
    else:
        print("警告: GP数据点数量不匹配!")
    
    # 检查最优点
    print("\n步骤8: 检查最优结果")
    print("-"*70)
    best = optimizer.max
    print(f"最优目标值: {best['target']:.4f}")
    print(f"最优参数:")
    for key, value in best['params'].items():
        print(f"  {key} = {value:.4f}")
    
    # 判断最优点是否来自LLM初始点
    best_index = None
    for i, res in enumerate(optimizer.res, 1):
        if res['target'] == best['target']:
            best_index = i
            break
    
    if best_index and best_index <= 5:
        print(f"\n最优点来自: LLM初始点 #{best_index}")
    else:
        print(f"\n最优点来自: BO迭代 #{best_index}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)
    
    # 返回验证结果
    return {
        'total_points': len(optimizer.space),
        'gp_train_points': len(optimizer._gp.X_train_),
        'llm_points_used': len(optimizer.space) >= 5,
        'integration_success': len(optimizer._gp.X_train_) == 7
    }


if __name__ == "__main__":
    result = test_warm_start_integration()
    
    print("\n最终验证:")
    print("-"*70)
    if result['integration_success']:
        print("状态: 成功")
        print("LLM Warm Start已正确集成到BO中")
        print("5个LLM初始点被BO接受并用于训练GP")
    else:
        print("状态: 失败")
        print("请检查集成代码")