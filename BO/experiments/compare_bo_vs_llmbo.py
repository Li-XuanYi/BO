"""
对比实验：传统BO vs LLMBO
目标：验证LLM Warm Start是否能提升优化效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt import BayesianOptimization
from llmbo import LLMBOOptimizer
from SPM import SPM
import numpy as np
import json
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


def run_traditional_bo(pbounds, init_points=5, n_iter=30, random_state=1):
    """
    运行传统BO
    """
    print("\n" + "="*70)
    print("运行传统BO（随机初始化）")
    print("="*70)
    
    start_time = time.time()
    
    optimizer = BayesianOptimization(
        f=charging_time_compute,
        pbounds=pbounds,
        random_state=random_state
    )
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    total_time = time.time() - start_time
    
    # 收集结果
    results = {
        'method': 'Traditional BO',
        'init_points': init_points,
        'n_iter': n_iter,
        'total_evaluations': len(optimizer.space),
        'total_time': total_time,
        'best_target': optimizer.max['target'],
        'best_params': optimizer.max['params'],
        'all_results': optimizer.res
    }
    
    print(f"\n传统BO完成:")
    print(f"  总评估次数: {results['total_evaluations']}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  最优充电步数: {-results['best_target']:.0f} 步")
    print(f"  最优参数:")
    for key, value in results['best_params'].items():
        print(f"    {key} = {value:.4f}")
    
    return results


def run_llmbo(pbounds, init_points=5, n_iter=30, llm_model="gpt-3.5-turbo", random_state=1):
    """
    运行LLMBO
    """
    print("\n" + "="*70)
    print("运行LLMBO（LLM Warm Start）")
    print("="*70)
    
    start_time = time.time()
    
    llmbo = LLMBOOptimizer(
        f=charging_time_compute,
        pbounds=pbounds,
        llm_model=llm_model,
        random_state=random_state
    )
    
    llmbo.maximize(
        init_points=init_points,
        n_iter=n_iter,
        use_llm_warm_start=True
    )
    
    total_time = time.time() - start_time
    
    # 收集结果
    results = {
        'method': 'LLMBO',
        'llm_model': llm_model,
        'init_points': init_points,
        'n_iter': n_iter,
        'total_evaluations': len(llmbo.optimizer.space),
        'total_time': total_time,
        'best_target': llmbo.max['target'],
        'best_params': llmbo.max['params'],
        'all_results': llmbo.res
    }
    
    print(f"\nLLMBO完成:")
    print(f"  总评估次数: {results['total_evaluations']}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  最优充电步数: {-results['best_target']:.0f} 步")
    print(f"  最优参数:")
    for key, value in results['best_params'].items():
        print(f"    {key} = {value:.4f}")
    
    return results


def analyze_results(bo_results, llmbo_results):
    """
    分析对比结果
    """
    print("\n" + "="*70)
    print("对比分析")
    print("="*70)
    
    # 提取前5个点的平均性能（初始点质量）
    bo_init_avg = np.mean([r['target'] for r in bo_results['all_results'][:5]])
    llmbo_init_avg = np.mean([r['target'] for r in llmbo_results['all_results'][:5]])
    
    print(f"\n1. 初始点质量对比（前5个点的平均目标值）:")
    print(f"   传统BO:  {bo_init_avg:.2f} (平均 {-bo_init_avg:.0f} 步)")
    print(f"   LLMBO:   {llmbo_init_avg:.2f} (平均 {-llmbo_init_avg:.0f} 步)")
    init_improvement = ((llmbo_init_avg - bo_init_avg) / abs(bo_init_avg)) * 100
    print(f"   改进:    {init_improvement:+.1f}%")
    
    print(f"\n2. 最终最优结果对比:")
    print(f"   传统BO:  {bo_results['best_target']:.2f} ({-bo_results['best_target']:.0f} 步)")
    print(f"   LLMBO:   {llmbo_results['best_target']:.2f} ({-llmbo_results['best_target']:.0f} 步)")
    final_improvement = ((llmbo_results['best_target'] - bo_results['best_target']) / abs(bo_results['best_target'])) * 100
    print(f"   改进:    {final_improvement:+.1f}%")
    
    print(f"\n3. 收敛速度对比（达到最优值的迭代次数）:")
    bo_best_iter = next(i for i, r in enumerate(bo_results['all_results'], 1) 
                        if r['target'] == bo_results['best_target'])
    llmbo_best_iter = next(i for i, r in enumerate(llmbo_results['all_results'], 1) 
                           if r['target'] == llmbo_results['best_target'])
    print(f"   传统BO:  第 {bo_best_iter} 次评估")
    print(f"   LLMBO:   第 {llmbo_best_iter} 次评估")
    
    print(f"\n4. 计算时间对比:")
    print(f"   传统BO:  {bo_results['total_time']:.2f} 秒")
    print(f"   LLMBO:   {llmbo_results['total_time']:.2f} 秒")
    print(f"   差异:    {llmbo_results['total_time'] - bo_results['total_time']:+.2f} 秒")
    
    # 绘制收敛曲线数据
    print(f"\n5. 收敛曲线数据:")
    print(f"\n   迭代次数  传统BO最优  LLMBO最优")
    print(f"   " + "-"*40)
    bo_best_so_far = []
    llmbo_best_so_far = []
    current_bo_best = float('-inf')
    current_llmbo_best = float('-inf')
    
    for i in range(min(len(bo_results['all_results']), len(llmbo_results['all_results']))):
        current_bo_best = max(current_bo_best, bo_results['all_results'][i]['target'])
        current_llmbo_best = max(current_llmbo_best, llmbo_results['all_results'][i]['target'])
        bo_best_so_far.append(current_bo_best)
        llmbo_best_so_far.append(current_llmbo_best)
        
        if (i + 1) % 5 == 0 or i < 5:  # 显示前5个和每5个
            print(f"   {i+1:^10}  {-current_bo_best:^11.0f}  {-current_llmbo_best:^11.0f}")
    
    return {
        'init_improvement': init_improvement,
        'final_improvement': final_improvement,
        'bo_best_iter': bo_best_iter,
        'llmbo_best_iter': llmbo_best_iter
    }


def save_results(bo_results, llmbo_results, comparison, filename=None):
    """
    保存结果到JSON文件
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.json"
    
    filepath = os.path.join('logs', filename)
    os.makedirs('logs', exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'traditional_bo': bo_results,
        'llmbo': llmbo_results,
        'comparison': comparison
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {filepath}")


def main():
    """
    主函数
    """
    print("="*70)
    print("BO vs LLMBO 对比实验")
    print("="*70)
    
    # 参数设置
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    init_points = 5
    n_iter = 10  # 先用小规模测试，确认无误后再改为30
    
    print(f"\n实验配置:")
    print(f"  初始探索点: {init_points}")
    print(f"  贝叶斯优化迭代: {n_iter}")
    print(f"  总评估次数: {init_points + n_iter}")
    print(f"  参数边界: {pbounds}")
    
    # 运行传统BO
    bo_results = run_traditional_bo(
        pbounds=pbounds,
        init_points=init_points,
        n_iter=n_iter,
        random_state=1
    )
    
    # 运行LLMBO
    llmbo_results = run_llmbo(
        pbounds=pbounds,
        init_points=init_points,
        n_iter=n_iter,
        llm_model="gpt-3.5-turbo",
        random_state=1
    )
    
    # 分析对比
    comparison = analyze_results(bo_results, llmbo_results)
    
    # 保存结果
    save_results(bo_results, llmbo_results, comparison)
    
    print("\n" + "="*70)
    print("实验完成")
    print("="*70)


if __name__ == "__main__":
    main()