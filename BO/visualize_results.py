"""
独立可视化模块
只负责接收充电参数并生成四幅对比图像
"""

import numpy as np
import matplotlib.pyplot as plt
from charging_utils import record_charging_process
from datetime import datetime


def plot_single_charging_curve(current1, charging_number, current2, 
                               method_name='Charging', color='#1f77b4',
                               save_path=None):
    """
    绘制单个充电策略的四合一图像
    
    参数:
        current1: 第一阶段充电电流 (A)
        charging_number: 第一阶段充电步数
        current2: 第二阶段充电电流 (A)
        method_name: 方法名称
        color: 曲线颜色
        save_path: 保存路径
    """
    print(f"\n正在记录充电过程: {method_name}")
    print(f"  current1 = {current1:.2f} A")
    print(f"  charging_number = {charging_number:.1f}")
    print(f"  current2 = {current2:.2f} A")
    
    # 记录充电过程
    data = record_charging_process(current1, charging_number, current2)
    
    # 创建四合一图像
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{method_name} - Charging Performance', 
                 fontsize=16, fontweight='bold')
    
    # 图1: SOC vs Time
    ax1 = axes[0, 0]
    ax1.plot(data['time'], data['soc'], 
             color=color, linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('SOC (%)', fontsize=12)
    ax1.set_title('State of Charge vs Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_ylim([0, 100])
    
    # 添加性能标注
    textstr = f'Total Time: {data["total_time"]:.1f} min\n'
    textstr += f'Total Steps: {data["total_steps"]}\n'
    textstr += f'Final SOC: {data["soc"][-1]:.1f}%'
    
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 图2: Voltage vs Time
    ax2 = axes[0, 1]
    ax2.plot(data['time'], data['voltage'], 
             color=color, linewidth=2.5, alpha=0.8)
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
    ax3.plot(data['time'], data['current'], 
             color=color, linewidth=2.5, alpha=0.8)
    ax3.set_xlabel('Time (min)', fontsize=12)
    ax3.set_ylabel('Current (A)', fontsize=12)
    ax3.set_title('Charging Current vs Time', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)
    
    # 图4: Temperature vs Time
    ax4 = axes[1, 1]
    ax4.plot(data['time'], data['temperature'], 
             color=color, linewidth=2.5, alpha=0.8)
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
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'{method_name}_{timestamp}.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存: {save_path}")
    
    plt.show()
    
    return data, fig


def plot_comparison(params1, params2, 
                   method1_name='Method 1', method2_name='Method 2',
                   color1='#1f77b4', color2='#d62728',
                   save_path=None):
    """
    绘制两个充电策略的对比图
    
    参数:
        params1: 第一个策略的参数 dict {'current1', 'charging_number', 'current2'}
        params2: 第二个策略的参数 dict {'current1', 'charging_number', 'current2'}
        method1_name: 第一个方法名称
        method2_name: 第二个方法名称
        color1: 第一个方法的颜色
        color2: 第二个方法的颜色
        save_path: 保存路径
    """
    print(f"\n正在生成对比图...")
    print(f"\n{method1_name}:")
    print(f"  current1 = {params1['current1']:.2f} A")
    print(f"  charging_number = {params1['charging_number']:.1f}")
    print(f"  current2 = {params1['current2']:.2f} A")
    
    print(f"\n{method2_name}:")
    print(f"  current1 = {params2['current1']:.2f} A")
    print(f"  charging_number = {params2['charging_number']:.1f}")
    print(f"  current2 = {params2['current2']:.2f} A")
    
    # 记录两个策略的充电过程
    data1 = record_charging_process(
        params1['current1'], 
        params1['charging_number'], 
        params1['current2']
    )
    
    data2 = record_charging_process(
        params2['current1'], 
        params2['charging_number'], 
        params2['current2']
    )
    
    # 创建四合一对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{method1_name} vs {method2_name} - Charging Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 图1: SOC vs Time
    ax1 = axes[0, 0]
    ax1.plot(data1['time'], data1['soc'], 
             color=color1, linewidth=2.5, 
             label=method1_name, alpha=0.8)
    ax1.plot(data2['time'], data2['soc'], 
             color=color2, linewidth=2.5, 
             label=method2_name, alpha=0.8)
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('SOC (%)', fontsize=12)
    ax1.set_title('State of Charge vs Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_xlim(left=0)
    ax1.set_ylim([0, 100])
    
    # 添加性能对比标注
    improvement = ((data1['total_steps'] - data2['total_steps']) / 
                   data1['total_steps']) * 100
    
    textstr = f'{method1_name}: {data1["total_time"]:.1f} min ({data1["total_steps"]} steps)\n'
    textstr += f'{method2_name}: {data2["total_time"]:.1f} min ({data2["total_steps"]} steps)\n'
    textstr += f'Improvement: {improvement:.1f}%'
    
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 图2: Voltage vs Time
    ax2 = axes[0, 1]
    ax2.plot(data1['time'], data1['voltage'], 
             color=color1, linewidth=2.5, 
             label=method1_name, alpha=0.8)
    ax2.plot(data2['time'], data2['voltage'], 
             color=color2, linewidth=2.5, 
             label=method2_name, alpha=0.8)
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
    ax3.plot(data1['time'], data1['current'], 
             color=color1, linewidth=2.5, 
             label=method1_name, alpha=0.8)
    ax3.plot(data2['time'], data2['current'], 
             color=color2, linewidth=2.5, 
             label=method2_name, alpha=0.8)
    ax3.set_xlabel('Time (min)', fontsize=12)
    ax3.set_ylabel('Current (A)', fontsize=12)
    ax3.set_title('Charging Current vs Time', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=11, loc='upper right')
    ax3.set_xlim(left=0)
    
    # 图4: Temperature vs Time
    ax4 = axes[1, 1]
    ax4.plot(data1['time'], data1['temperature'], 
             color=color1, linewidth=2.5, 
             label=method1_name, alpha=0.8)
    ax4.plot(data2['time'], data2['temperature'], 
             color=color2, linewidth=2.5, 
             label=method2_name, alpha=0.8)
    ax4.axhline(y=38, color='red', linestyle='--', linewidth=1.5, 
                label='Temp Limit (38°C)', alpha=0.7)
    ax4.set_xlabel('Time (min)', fontsize=12)
    ax4.set_ylabel('Temperature (°C)', fontsize=12)
    ax4.set_title('Cell Temperature vs Time', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10, loc='upper left')
    ax4.set_xlim(left=0)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'Comparison_{timestamp}.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图像已保存: {save_path}")
    
    plt.show()
    
    return {'data1': data1, 'data2': data2, 'fig': fig}


if __name__ == "__main__":
    print("="*70)
    print("可视化模块 - 手动输入参数接口")
    print("="*70)
    
    # ====================================================================
    # 手动输入参数区域 - 您可以在这里修改参数
    # ====================================================================
    
    # 方式1: 绘制单个充电策略
    USE_SINGLE_PLOT = False  # 改为True使用单图模式
    
    if USE_SINGLE_PLOT:
        # 单个策略参数
        current1 = 5.0
        charging_number = 15
        current2 = 3.0
        
        plot_single_charging_curve(
            current1=current1,
            charging_number=charging_number,
            current2=current2,
            method_name='My_Strategy',
            color='#1f77b4'
        )
    
    # 方式2: 绘制两个策略的对比图
    else:
        # 第一个策略参数 (例如: BO的结果)
        params_method1 = {
            'current1': 4.92,
            'charging_number': 13.8,
            'current2': 4.0
        }
        
        # 第二个策略参数 (例如: LLMBO的结果)
        params_method2 = {
            'current1': 5.2,
            'charging_number': 12.5,
            'current2': 3.8
        }
        
        plot_comparison(
            params1=params_method1,
            params2=params_method2,
            method1_name='Traditional BO',
            method2_name='LLMBO',
            color1='#1f77b4',  # 蓝色
            color2='#d62728'   # 红色
        )
    
    print("\n" + "="*70)
    print("可视化完成!")
    print("="*70)
