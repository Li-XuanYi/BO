"""
完整对比实验: BO vs LLMBO
自动记录数据并生成可视化图像
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bayes_opt import BayesianOptimization
from llmbo import LLMBOOptimizer
from SPM import SPM
import numpy as np
import matplotlib.pyplot as plt
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


def record_charging_process(current1, charging_number, current2):
    """记录充电过程的完整时间序列数据"""
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
        soc_history.append(env.soc * 100)
        voltage_history.append(env.voltage)
        current_history.append(current)
        temp_history.append(env.temp - 273.15)
        
        _, done, _ = env.step(current)
        i += 1
        t += env.sett['sample_time'] / 60
        
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


def run_traditional_bo(pbounds, init_points=5, n_iter=10):
    """运行传统BO"""
    print("\n" + "="*70)
    print("运行传统BO (随机初始化)")
    print("="*70)
    
    start_time = time.time()
    
    optimizer = BayesianOptimization(
        f=charging_time_compute,
        pbounds=pbounds,
        random_state=42
    )
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    total_time = time.time() - start_time
    
    # 记录最优参数的充电过程
    best_params = optimizer.max['params']
    print(f"\n记录BO最优参数的充电过程...")
    charging_data = record_charging_process(
        current1=best_params['current1'],
        charging_number=best_params['charging_number'],
        current2=best_params['current2']
    )
    
    results = {
        'method': 'Traditional BO',
        'best_target': optimizer.max['target'],
        'best_params': optimizer.max['params'],
        'charging_data': charging_data,
        'total_time': total_time,
        'color': '#1f77b4'  # 蓝色
    }
    
    print(f"\nBO完成:")
    print(f"  最优充电步数: {-results['best_target']:.0f} 步")
    print(f"  总耗时: {total_time:.2f} 秒")
    
    return results


def run_llmbo(pbounds, init_points=5, n_iter=10):
    """运行LLMBO"""
    print("\n" + "="*70)
    print("运行LLMBO (LLM三大策略)")
    print("="*70)
    
    start_time = time.time()
    
    llmbo = LLMBOOptimizer(
        f=charging_time_compute,
        pbounds=pbounds,
        llm_model="gpt-3.5-turbo",
        random_state=42,
        use_enhanced_kernel=True,
        use_dynamic_sampling=True
    )
    
    result = llmbo.maximize(
        init_points=init_points,
        n_iter=n_iter,
        use_llm_warm_start=True,
        record_best=True  # 自动记录最优结果
    )
    
    results = {
        'method': 'LLMBO (Ours)',
        'best_target': result['optimizer_result']['target'],
        'best_params': result['optimizer_result']['params'],
        'charging_data': result['charging_data'],
        'total_time': result['total_time'],
        'gamma_history': result['gamma_history'],
        'color': '#d62728'  # 红色
    }
    
    print(f"\nLLMBO完成:")
    print(f"  最优充电步数: {-results['best_target']:.0f} 步")
    print(f"  总耗时: {results['total_time']:.2f} 秒")
    
    return results


def plot_comparison(bo_results, llmbo_results, save_path='comparison.png'):
    """生成四合一对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BO vs LLMBO - Charging Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    bo_data = bo_results['charging_data']
    llmbo_data = llmbo_results['charging_data']
    
    # 图1: SOC vs Time
    ax1 = axes[0, 0]
    ax1.plot(bo_data['time'], bo_data['soc'], 
             color=bo_results['color'], linewidth=2.5, 
             label=bo_results['method'], alpha=0.8)
    ax1.plot(llmbo_data['time'], llmbo_data['soc'], 
             color=llmbo_results['color'], linewidth=2.5, 
             label=llmbo_results['method'], alpha=0.8)
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('SOC (%)', fontsize=12)
    ax1.set_title('State of Charge vs Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_xlim(left=0)
    ax1.set_ylim([0, 100])
    
    # 添加性能标注
    improvement = ((bo_data['total_steps'] - llmbo_data['total_steps']) / 
                   bo_data['total_steps']) * 100
    
    textstr = f'{bo_results["method"]}: {bo_data["total_time"]:.1f} min ({bo_data["total_steps"]} steps)\n'
    textstr += f'{llmbo_results["method"]}: {llmbo_data["total_time"]:.1f} min ({llmbo_data["total_steps"]} steps)\n'
    textstr += f'Improvement: {improvement:.1f}%'
    
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 图2: Voltage vs Time
    ax2 = axes[0, 1]
    ax2.plot(bo_data['time'], bo_data['voltage'], 
             color=bo_results['color'], linewidth=2.5, 
             label=bo_results['method'], alpha=0.8)
    ax2.plot(llmbo_data['time'], llmbo_data['voltage'], 
             color=llmbo_results['color'], linewidth=2.5, 
             label=llmbo_results['method'], alpha=0.8)
    ax2.axhline(y=4.2, color='red', linestyle='--', linewidth=1.5, 
                label='Voltage Limit (4.2V)', alpha=0.7)
    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('Voltage (V)', fontsize=12)
    ax2.set_title('Terminal Voltage vs Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlim(left=0)
    
    # 图3: Current vs Time
    ax3 = axes[1, 0]
    ax3.plot(bo_data['time'], bo_data['current'], 
             color=bo_results['color'], linewidth=2.5, 
             label=bo_results['method'], alpha=0.8)
    ax3.plot(llmbo_data['time'], llmbo_data['current'], 
             color=llmbo_results['color'], linewidth=2.5, 
             label=llmbo_results['method'], alpha=0.8)
    ax3.set_xlabel('Time (min)', fontsize=12)
    ax3.set_ylabel('Current (A)', fontsize=12)
    ax3.set_title('Charging Current vs Time', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=11, loc='upper right')
    ax3.set_xlim(left=0)
    
    # 图4: Temperature vs Time
    ax4 = axes[1, 1]
    ax4.plot(bo_data['time'], bo_data['temperature'], 
             color=bo_results['color'], linewidth=2.5, 
             label=bo_results['method'], alpha=0.8)
    ax4.plot(llmbo_data['time'], llmbo_data['temperature'], 
             color=llmbo_results['color'], linewidth=2.5, 
             label=llmbo_results['method'], alpha=0.8)
    ax4.axhline(y=36, color='red', linestyle='--', linewidth=1.5, 
                label='Temp Limit (36°C)', alpha=0.7)
    ax4.set_xlabel('Time (min)', fontsize=12)
    ax4.set_ylabel('Temperature (°C)', fontsize=12)
    ax4.set_title('Cell Temperature vs Time', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10, loc='upper left')
    ax4.set_xlim(left=0)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存: {save_path}")
    
    plt.show()
    
    return fig


def main():
    """主函数"""
    print("="*70)
    print("BO vs LLMBO 完整对比实验")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    init_points = 5
    n_iter = 10
    
    print(f"\n实验配置:")
    print(f"  初始点数: {init_points}")
    print(f"  迭代次数: {n_iter}")
    print(f"  总评估: {init_points + n_iter} 次/方法")
    
    # 运行传统BO
    bo_results = run_traditional_bo(pbounds, init_points, n_iter)
    
    # 运行LLMBO
    llmbo_results = run_llmbo(pbounds, init_points, n_iter)
    
    # 对比分析
    print("\n" + "="*70)
    print("对比分析")
    print("="*70)
    
    improvement = ((bo_results['best_target'] - llmbo_results['best_target']) / 
                   abs(bo_results['best_target'])) * 100
    
    print(f"\n充电步数:")
    print(f"  BO:    {-bo_results['best_target']:.0f} 步")
    print(f"  LLMBO: {-llmbo_results['best_target']:.0f} 步")
    print(f"  改进:  {improvement:+.1f}%")
    
    time_improvement = ((bo_results['charging_data']['total_time'] - 
                        llmbo_results['charging_data']['total_time']) / 
                       bo_results['charging_data']['total_time']) * 100
    
    print(f"\n充电时间:")
    print(f"  BO:    {bo_results['charging_data']['total_time']:.1f} 分钟")
    print(f"  LLMBO: {llmbo_results['charging_data']['total_time']:.1f} 分钟")
    print(f"  改进:  {time_improvement:+.1f}%")
    
    print(f"\n计算时间:")
    print(f"  BO:    {bo_results['total_time']:.1f} 秒")
    print(f"  LLMBO: {llmbo_results['total_time']:.1f} 秒")
    
    # 生成可视化
    print("\n" + "="*70)
    print("生成对比图像...")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'BO_vs_LLMBO_{timestamp}.png'
    
    plot_comparison(bo_results, llmbo_results, save_path)
    
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()