"""
主运行文件: 运行BO和LLMBO对比实验
"""

import sys
import os
import time
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bayes_opt import BayesianOptimization
from charging_utils import charging_time_compute, record_charging_process
from visualize_results import plot_comparison


def run_traditional_bo(pbounds, init_points=5, n_iter=10, random_state=42):
    """
    运行传统BO
    
    参数:
        pbounds: 参数边界
        init_points: 初始随机点数
        n_iter: 迭代次数
        random_state: 随机种子
    
    返回:
        dict: 优化结果
    """
    print("\n" + "="*70)
    print("运行传统BO (随机初始化)")
    print("="*70)
    
    start_time = time.time()
    
    optimizer = BayesianOptimization(
        f=charging_time_compute,
        pbounds=pbounds,
        random_state=random_state
    )
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    total_time = time.time() - start_time
    
    # 获取最优参数
    best_params = optimizer.max['params']
    
    print(f"\n传统BO完成:")
    print(f"  总评估次数: {len(optimizer.space)}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  最优充电步数: {-optimizer.max['target']:.0f} 步")
    print(f"  最优参数:")
    print(f"    current1 = {best_params['current1']:.4f} A")
    print(f"    charging_number = {best_params['charging_number']:.4f}")
    print(f"    current2 = {best_params['current2']:.4f} A")
    
    # 记录最优参数的充电过程
    print(f"\n记录最优参数的充电过程...")
    charging_data = record_charging_process(
        current1=best_params['current1'],
        charging_number=best_params['charging_number'],
        current2=best_params['current2']
    )
    
    results = {
        'method': 'Traditional BO',
        'total_evaluations': len(optimizer.space),
        'total_time': total_time,
        'best_target': optimizer.max['target'],
        'best_params': best_params,
        'charging_data': {
            'total_steps': int(charging_data['total_steps']),
            'total_time': float(charging_data['total_time'])
        },
        'all_results': [
            {
                'target': res['target'],
                'params': res['params']
            }
            for res in optimizer.res
        ]
    }
    
    return results


def run_llmbo(pbounds, init_points=5, n_iter=10, random_state=42):
    """
    运行LLMBO (暂时使用传统BO,后续会替换为真正的LLMBO)
    
    参数:
        pbounds: 参数边界
        init_points: 初始点数
        n_iter: 迭代次数
        random_state: 随机种子
    
    返回:
        dict: 优化结果
    """
    print("\n" + "="*70)
    print("运行LLMBO (当前版本: 使用BO占位)")
    print("="*70)
    print("注意: 这是临时版本,后续会替换为真正的LLMBO实现")
    
    start_time = time.time()
    
    # 暂时使用传统BO作为占位符
    # TODO: 替换为真正的LLMBO实现
    optimizer = BayesianOptimization(
        f=charging_time_compute,
        pbounds=pbounds,
        random_state=random_state + 1  # 使用不同的种子以获得不同结果
    )
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    total_time = time.time() - start_time
    
    # 获取最优参数
    best_params = optimizer.max['params']
    
    print(f"\nLLMBO完成:")
    print(f"  总评估次数: {len(optimizer.space)}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  最优充电步数: {-optimizer.max['target']:.0f} 步")
    print(f"  最优参数:")
    print(f"    current1 = {best_params['current1']:.4f} A")
    print(f"    charging_number = {best_params['charging_number']:.4f}")
    print(f"    current2 = {best_params['current2']:.4f} A")
    
    # 记录最优参数的充电过程
    print(f"\n记录最优参数的充电过程...")
    charging_data = record_charging_process(
        current1=best_params['current1'],
        charging_number=best_params['charging_number'],
        current2=best_params['current2']
    )
    
    results = {
        'method': 'LLMBO',
        'total_evaluations': len(optimizer.space),
        'total_time': total_time,
        'best_target': optimizer.max['target'],
        'best_params': best_params,
        'charging_data': {
            'total_steps': int(charging_data['total_steps']),
            'total_time': float(charging_data['total_time'])
        },
        'all_results': [
            {
                'target': res['target'],
                'params': res['params']
            }
            for res in optimizer.res
        ]
    }
    
    return results


def save_results(bo_results, llmbo_results, save_dir='./results'):
    """
    保存实验结果到JSON文件
    
    参数:
        bo_results: BO结果
        llmbo_results: LLMBO结果
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'comparison_results_{timestamp}.json')
    
    # 准备保存的数据
    results = {
        'timestamp': datetime.now().isoformat(),
        'traditional_bo': bo_results,
        'llmbo': llmbo_results,
        'comparison': {
            'bo_best_steps': int(-bo_results['best_target']),
            'llmbo_best_steps': int(-llmbo_results['best_target']),
            'improvement_steps': int(-bo_results['best_target'] - (-llmbo_results['best_target'])),
            'improvement_percent': float(
                ((-bo_results['best_target']) - (-llmbo_results['best_target'])) / 
                (-bo_results['best_target']) * 100
            )
        }
    }
    
    # 保存到JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {save_path}")
    
    return save_path


def main():
    """主函数"""
    print("="*70)
    print("BO vs LLMBO 完整对比实验")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ====================================================================
    # 实验配置 - 您可以在这里修改参数
    # ====================================================================
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    init_points = 5   # 初始随机点数
    n_iter = 10       # 迭代次数
    random_state = 42  # 随机种子
    
    print(f"\n实验配置:")
    print(f"  参数范围:")
    print(f"    current1: {pbounds['current1']}")
    print(f"    charging_number: {pbounds['charging_number']}")
    print(f"    current2: {pbounds['current2']}")
    print(f"  初始点数: {init_points}")
    print(f"  迭代次数: {n_iter}")
    print(f"  总评估: {init_points + n_iter} 次/方法")
    print(f"  随机种子: {random_state}")
    
    # 运行传统BO
    bo_results = run_traditional_bo(pbounds, init_points, n_iter, random_state)
    
    # 运行LLMBO
    llmbo_results = run_llmbo(pbounds, init_points, n_iter, random_state)
    
    # 对比分析
    print("\n" + "="*70)
    print("对比分析")
    print("="*70)
    
    bo_steps = -bo_results['best_target']
    llmbo_steps = -llmbo_results['best_target']
    improvement = ((bo_steps - llmbo_steps) / bo_steps) * 100
    
    print(f"\n充电步数:")
    print(f"  BO:    {bo_steps:.0f} 步")
    print(f"  LLMBO: {llmbo_steps:.0f} 步")
    print(f"  改进:  {improvement:+.1f}%")
    
    time_improvement = ((bo_results['charging_data']['total_time'] - 
                        llmbo_results['charging_data']['total_time']) / 
                       bo_results['charging_data']['total_time']) * 100
    
    print(f"\n充电时间:")
    print(f"  BO:    {bo_results['charging_data']['total_time']:.2f} 分钟")
    print(f"  LLMBO: {llmbo_results['charging_data']['total_time']:.2f} 分钟")
    print(f"  改进:  {time_improvement:+.1f}%")
    
    print(f"\n计算时间:")
    print(f"  BO:    {bo_results['total_time']:.2f} 秒")
    print(f"  LLMBO: {llmbo_results['total_time']:.2f} 秒")
    
    # 保存结果
    print("\n" + "="*70)
    print("保存实验结果...")
    print("="*70)
    save_results(bo_results, llmbo_results)
    
    # 生成可视化
    print("\n" + "="*70)
    print("生成对比图像...")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'./results/BO_vs_LLMBO_{timestamp}.png'
    
    plot_comparison(
        params1=bo_results['best_params'],
        params2=llmbo_results['best_params'],
        method1_name='Traditional BO',
        method2_name='LLMBO',
        color1='#1f77b4',
        color2='#d62728',
        save_path=save_path
    )
    
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
