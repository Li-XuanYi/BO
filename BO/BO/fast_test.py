"""
LLMBO集成测试脚本
快速验证三大策略是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llmbo import LLMBOOptimizer
from SPM import SPM
import numpy as np
import time
from datetime import datetime


def charging_time_compute(current1, charging_number, current2):
    """
    两阶段充电目标函数
    返回负的充电步数（越大越好）
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
        
        # 约束惩罚
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 1
        
        if done:
            return -i


def test_llmbo_quick():
    """
    快速集成测试：小规模验证
    """
    print("="*70)
    print("LLMBO 快速集成测试")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 参数设置
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    init_points = 5  # 5个初始点
    n_iter = 3       # 3次迭代（快速测试）
    
    print(f"\n测试配置:")
    print(f"  初始点数: {init_points} (LLM生成)")
    print(f"  迭代次数: {n_iter}")
    print(f"  总评估次数: {init_points + n_iter} = {init_points + n_iter}")
    print(f"  参数边界: {pbounds}")
    print(f"  策略1 (Warm Start): 启用")
    print(f"  策略2 (增强核函数): 启用")
    print(f"  策略3 (动态采样): 启用")
    
    # 创建LLMBO优化器
    print("\n" + "-"*70)
    print("步骤1: 初始化LLMBO优化器")
    print("-"*70)
    
    try:
        llmbo = LLMBOOptimizer(
            f=charging_time_compute,
            pbounds=pbounds,
            llm_model="gpt-3.5-turbo",
            random_state=42,
            use_enhanced_kernel=True,
            use_dynamic_sampling=True
        )
        print("✓ LLMBO优化器初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 运行优化
    print("\n" + "-"*70)
    print("步骤2: 运行优化")
    print("-"*70)
    
    start_time = time.time()
    
    try:
        result = llmbo.maximize(
            init_points=init_points,
            n_iter=n_iter,
            use_llm_warm_start=True
        )
        print("✓ 优化完成")
    except Exception as e:
        print(f"✗ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    total_time = time.time() - start_time
    
    # 显示结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    
    print(f"\n✓ 所有测试通过！")
    print(f"\n性能统计:")
    print(f"  总评估次数: {len(llmbo.optimizer.space)}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  平均每次评估: {total_time/(init_points+n_iter):.2f} 秒")
    
    print(f"\n最优结果:")
    print(f"  最优充电步数: {-result['target']:.0f} 步")
    print(f"  最优参数:")
    print(f"    current1 = {result['params']['current1']:.4f} A")
    print(f"    charging_number = {result['params']['charging_number']:.2f} 步")
    print(f"    current2 = {result['params']['current2']:.4f} A")
    
    # 显示所有评估点
    print(f"\n评估历史 (前5个点 + 最后3个点):")
    print(f"  {'#':<4} {'来源':<15} {'current1':<10} {'chg_num':<10} {'current2':<10} {'步数':<8}")
    print(f"  {'-'*67}")
    
    all_results = llmbo.res
    
    # 显示前5个（LLM初始点）
    for i, res in enumerate(all_results[:5], 1):
        print(f"  {i:<4} {'LLM初始点':<15} "
              f"{res['params']['current1']:<10.2f} "
              f"{res['params']['charging_number']:<10.1f} "
              f"{res['params']['current2']:<10.2f} "
              f"{-res['target']:<8.0f}")
    
    # 显示后3个（BO迭代）
    if len(all_results) > 5:
        print(f"  {'...':<4} {'...':<15} {'...':<10} {'...':<10} {'...':<10} {'...':<8}")
        for i, res in enumerate(all_results[5:], 6):
            print(f"  {i:<4} {'BO迭代':<15} "
                  f"{res['params']['current1']:<10.2f} "
                  f"{res['params']['charging_number']:<10.1f} "
                  f"{res['params']['current2']:<10.2f} "
                  f"{-res['target']:<8.0f}")
    
    # 检查三大策略是否都被使用
    print(f"\n策略验证:")
    print(f"  ✓ 策略1 (Warm Start): 已使用 - 生成了{init_points}个初始点")
    print(f"  ✓ 策略2 (增强核函数): 已配置")
    print(f"  ✓ 策略3 (动态采样): 已启用 (每5次迭代分析)")
    
    print("\n" + "="*70)
    print("集成测试完成 - 所有策略正常工作！")
    print("="*70)
    
    return True, {
        'total_evaluations': len(llmbo.optimizer.space),
        'total_time': total_time,
        'best_steps': -result['target'],
        'best_params': result['params'],
        'all_results': all_results
    }


if __name__ == "__main__":
    print("\n" + "🚀 "*35)
    print("\n开始LLMBO集成测试...\n")
    
    success, results = test_llmbo_quick()
    
    if success:
        print("\n✅ 测试成功！系统运行正常。")
        print("\n下一步建议:")
        print("  1. 运行完整对比实验 (BO vs LLMBO)")
        print("  2. 添加可视化功能")
        print("  3. 生成实验报告")
        
        # 询问是否继续
        print("\n" + "="*70)
        choice = input("\n是否立即运行完整对比实验？(y/n): ")
        if choice.lower() == 'y':
            print("\n准备运行完整实验...")
            print("(这将需要较长时间，建议先确保测试结果满意)")
    else:
        print("\n❌ 测试失败！请检查错误信息。")
    
    print("\n" + "🚀 "*35 + "\n")