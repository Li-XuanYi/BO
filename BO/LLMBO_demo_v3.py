"""
LLMBO Demo V3 - 聚焦核心创新

论文核心方法（manuscript1.pdf）:
1. LLM Warm Start (方程4) - 智能初始点
2. LLM Enhanced Kernel (方程5-7) - 增强核函数 + 自适应gamma
3. LLM Enhanced Sampling (方程8-9) - ⭐核心创新！智能候选采样

与BO_demo.py的区别:
- BO_demo: 传统贝叶斯优化，随机初始化，标准采样
- LLMBO_demo: LLM增强，智能初始化，智能采样策略

使用说明:
1. 运行此文件进行LLMBO优化
2. 结果可以手动输入到visualize_results.py进行可视化
3. 不需要同时运行BO和LLMBO

重要提示:
- 避免了实时梯度计算（之前导致极慢）
- 聚焦LLM智能决策（论文核心）
- 保持计算效率
"""

from llmbo.llmbo_optimizer_v3 import LLMBOOptimizerV3
from SPM import SPM
import numpy as np
import json
from datetime import datetime


def charging_time_compute(current1, charging_number, current2):
    """
    两阶段充电目标函数
    
    参数:
        current1: 第一阶段电流 (A)
        charging_number: 第一阶段持续步数
        current2: 第二阶段电流 (A)
    
    返回:
        负的充电步数（BO求最大值，所以用负数）
    """
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    while not done:
        # 两阶段充电逻辑
        if i < int(charging_number):
            # 第一阶段
            current = current1
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        else:
            # 第二阶段
            current = current2
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        
        # 执行一步
        _, done, _ = env.step(current)
        i += 1
        
        # 约束惩罚
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 1
        
        if done:
            return -i  # 返回负值（最小化充电步数）


def save_results_to_file(result, history, filename="llmbo_results.json"):
    """
    保存结果到文件，方便后续可视化
    
    参数:
        result: 最优结果
        history: 优化历史
        filename: 输出文件名
    """
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'method': 'LLMBO',
        'best_result': {
            'charging_steps': -result['target'],
            'current1': result['params']['current1'],
            'charging_number': result['params']['charging_number'],
            'current2': result['params']['current2']
        },
        'all_evaluations': [
            {
                'iteration': eval['iteration'],
                'charging_steps': eval['charging_steps'],
                'current1': eval['params']['current1'],
                'charging_number': eval['params']['charging_number'],
                'current2': eval['params']['current2']
            }
            for eval in history['evaluations']
        ],
        'convergence': [
            {
                'iteration': conv['iteration'],
                'best_charging_steps': conv['best_charging_steps']
            }
            for conv in history['best_per_iteration']
        ]
    }
    
    # 添加gamma历史（如果有）
    if 'gamma_history' in history:
        output_data['gamma_history'] = history['gamma_history']
    
    # 添加采样统计（如果有）
    if 'sampling_stats' in history:
        output_data['sampling_stats'] = history['sampling_stats']
    
    # 保存到文件
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n结果已保存到: {filename}")
    print("可以将此结果手动输入到visualize_results.py进行可视化")


def print_results_summary(result, history):
    """打印结果摘要"""
    print("\n" + "="*70)
    print("LLMBO优化结果摘要")
    print("="*70)
    
    # 最优结果
    print(f"\n【最优结果】")
    print(f"  充电步数: {-result['target']:.0f} 步")
    print(f"  参数:")
    print(f"    current1 = {result['params']['current1']:.4f} A")
    print(f"    charging_number = {result['params']['charging_number']:.2f} 步")
    print(f"    current2 = {result['params']['current2']:.4f} A")
    
    # 优化统计
    print(f"\n【优化统计】")
    print(f"  总评估次数: {len(history['evaluations'])}")
    initial_best = history['best_per_iteration'][0]['best_charging_steps']
    final_best = history['best_per_iteration'][-1]['best_charging_steps']
    improvement = initial_best - final_best
    improvement_pct = (improvement / initial_best) * 100
    print(f"  初始最优: {initial_best:.0f} 步")
    print(f"  最终最优: {final_best:.0f} 步")
    print(f"  改进: {improvement:.0f} 步 ({improvement_pct:.1f}%)")
    
    # LLM参与统计（如果有）
    if 'gamma_history' in history:
        print(f"\n【LLM增强核函数】")
        gamma_initial = history['gamma_history'][0]
        gamma_final = history['gamma_history'][-1]
        print(f"  初始γ: {gamma_initial:.4f}")
        print(f"  最终γ: {gamma_final:.4f}")
        print(f"  γ调整次数: {len(history['gamma_history'])-1}")
    
    if 'sampling_stats' in history:
        print(f"\n【LLM增强采样】")
        stats = history['sampling_stats']
        print(f"  最终策略: {stats['current_strategy']}")
        print(f"  跟踪参数: {stats['n_parameters_tracked']}个")
        if stats['n_parameters_tracked'] > 0:
            print(f"  聚焦均值μ: {stats['mu_values']}")
            print(f"  聚焦标准差σ: {stats['sigma_values']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("="*70)
    print("LLMBO Demo V3 - LLM增强贝叶斯优化")
    print("聚焦核心创新：LLM智能采样策略")
    print("="*70)
    
    # 参数边界定义
    pbounds = {
        "current1": (3, 6),           # 第一阶段充电电流 (A)
        "charging_number": (5, 25),   # 第一阶段持续步数
        "current2": (1, 4)            # 第二阶段充电电流 (A)
    }
    
    print("\n【问题设置】")
    print(f"  优化目标: 最小化充电步数")
    print(f"  参数空间:")
    for param, (low, high) in pbounds.items():
        print(f"    {param}: [{low}, {high}]")
    
    # 创建LLMBO优化器V3
    print("\n【初始化LLMBO优化器V3】")
    print("  核心策略:")
    print("    1. LLM Warm Start - 智能初始点生成")
    print("    2. LLM Enhanced Kernel - 增强核函数（启发式梯度）")
    print("    3. ⭐ LLM Enhanced Sampling - 智能候选采样（论文核心！）")
    print("\n  关键改进:")
    print("    ✅ 避免实时SPM梯度计算（保持效率）")
    print("    ✅ 聚焦LLM智能决策")
    print("    ✅ 三策略协同工作")
    
    llmbo = LLMBOOptimizerV3(
        f=charging_time_compute,
        pbounds=pbounds,
        llm_model="gpt-3.5-turbo",
        random_state=None,              # None表示真随机（避免重复）
        use_warm_start=True,            # 启用LLM Warm Start
        use_enhanced_kernel=True,       # 启用增强核函数
        use_enhanced_sampling=True      # 启用LLM增强采样（核心！）
    )
    
    # 运行优化
    print("\n" + "="*70)
    print("开始优化")
    print("="*70)
    print("\n注意: 每次SPM评估约需1-2秒")
    print("      整个优化过程约需5-10分钟\n")
    
    result = llmbo.maximize(
        init_points=5,                  # 5个LLM智能初始点
        n_iter=30,                      # 30次贝叶斯优化迭代
        verbose=2                       # 详细输出
    )
    
    # 获取优化历史
    history = llmbo.get_optimization_history()
    
    # 打印结果摘要
    print_results_summary(result, history)
    
    # 保存结果到文件
    save_results_to_file(result, history, "llmbo_results.json")
    
    print("\n" + "="*70)
    print("运行完成")
    print("="*70)
    
    print("\n【下一步】")
    print("  1. 查看保存的结果文件: llmbo_results.json")
    print("  2. 将结果手动输入到visualize_results.py进行可视化")
    print("  3. 对比BO_demo的结果，验证LLMBO的优势")
    
    print("\n【核心创新验证】")
    if 'sampling_stats' in history:
        stats = history['sampling_stats']
        if stats['current_strategy'] != 'BALANCED':
            print(f"  ✅ LLM智能调整采样策略: {stats['current_strategy']}")
        if stats['n_parameters_tracked'] > 0:
            print(f"  ✅ LLM识别敏感参数: {stats['n_parameters_tracked']}个")
    else:
        print("  ℹ️ LLM采样策略未能完全启动，可能需要更多迭代")
    
    if 'gamma_history' in history:
        gamma_range = max(history['gamma_history']) - min(history['gamma_history'])
        if gamma_range > 0.05:
            print(f"  ✅ LLM智能调整耦合强度γ: 变化范围{gamma_range:.4f}")
    
    print("\n" + "="*70)
