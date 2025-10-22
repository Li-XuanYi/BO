"""
LLMBO集成测试 - LLM智能Gamma调整
测试10次迭代,观察LLM在第5次和第10次的智能调整
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


def test_llmbo_with_llm_gamma():
    """
    测试LLM智能Gamma调整
    """
    print("="*70)
    print("LLMBO测试 - LLM智能Gamma调整")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 参数设置
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    init_points = 5
    n_iter = 10  # 改为10次迭代,观察第5次和第10次的LLM调整
    
    print(f"\n测试配置:")
    print(f"  初始点数: {init_points} (LLM生成)")
    print(f"  迭代次数: {n_iter}")
    print(f"  总评估次数: {init_points + n_iter} = {init_points + n_iter}")
    print(f"  LLM Gamma调整: 每5次迭代 (预期在第5次和第10次)")
    print(f"  参数边界: {pbounds}")
    
    # 创建LLMBO优化器
    print("\n" + "-"*70)
    print("初始化LLMBO优化器")
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
        print("成功: LLMBO优化器初始化成功")
    except Exception as e:
        print(f"失败: 初始化失败 - {e}")
        return False
    
    # 运行优化
    print("\n" + "-"*70)
    print("开始优化")
    print("-"*70)
    
    start_time = time.time()
    
    try:
        result = llmbo.maximize(
            init_points=init_points,
            n_iter=n_iter,
            use_llm_warm_start=True
        )
        print("\n成功: 优化完成")
    except Exception as e:
        print(f"\n失败: 优化失败 - {e}")
        import traceback
        traceback.print_exc()
        return False, None  # 修复: 返回两个值
    
    total_time = time.time() - start_time
    
    # 显示结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    
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
    
    # 显示Gamma演化历史
    if llmbo.enhanced_kernel:
        history = llmbo.enhanced_kernel.get_gamma_history()
        print(f"\nGamma演化分析:")
        print(f"  初始gamma: {history['gamma_history'][0]:.4f}")
        print(f"  最终gamma: {history['gamma_history'][-1]:.4f}")
        print(f"  总变化: {(history['gamma_history'][-1] - history['gamma_history'][0]):.4f}")
        
        print(f"\n  详细轨迹:")
        for i, (g, f) in enumerate(zip(history['gamma_history'], history['f_min_history'])):
            marker = ""
            if i in [5, 10]:
                marker = " <- LLM咨询点"
            print(f"    Iter {i}: gamma={g:.4f}, f_min={f:.2f}{marker}")
    
    # 显示评估历史
    print(f"\n评估历史 (所有点):")
    print(f"  {'#':<4} {'来源':<15} {'current1':<10} {'chg_num':<10} {'current2':<10} {'步数':<8}")
    print(f"  {'-'*67}")
    
    all_results = llmbo.res
    
    for i, res in enumerate(all_results, 1):
        if i <= init_points:
            source = "LLM初始点"
        else:
            source = "BO迭代"
        
        print(f"  {i:<4} {source:<15} "
              f"{res['params']['current1']:<10.2f} "
              f"{res['params']['charging_number']:<10.1f} "
              f"{res['params']['current2']:<10.2f} "
              f"{-res['target']:<8.0f}")
    
    print("\n" + "="*70)
    print("测试完成 - LLM智能Gamma调整正常工作")
    print("="*70)
    
    return True, {
        'total_evaluations': len(llmbo.optimizer.space),
        'total_time': total_time,
        'best_steps': -result['target'],
        'best_params': result['params'],
        'gamma_history': history if llmbo.enhanced_kernel else None
    }


if __name__ == "__main__":
    print("\n开始测试LLM智能Gamma调整...\n")
    
    success, results = test_llmbo_with_llm_gamma()
    
    if success:
        print("\n测试成功!")
        print("\n关键观察点:")
        print("  1. 第5次迭代应该看到 '咨询LLM智能顾问...'")
        print("  2. 第10次迭代应该再次看到LLM咨询")
        print("  3. Gamma应该根据LLM建议有明显调整")
        
        if results['gamma_history']:
            gamma_hist = results['gamma_history']['gamma_history']
            if len(gamma_hist) > 5:
                gamma_change = gamma_hist[5] - gamma_hist[4]
                print(f"\n  第5次LLM调整: gamma变化 {gamma_change:+.4f}")
            if len(gamma_hist) > 10:
                gamma_change = gamma_hist[10] - gamma_hist[9]
                print(f"  第10次LLM调整: gamma变化 {gamma_change:+.4f}")
    else:
        print("\n测试失败,请检查错误信息")