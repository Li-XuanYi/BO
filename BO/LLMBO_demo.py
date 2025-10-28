"""
LLMBO Demo - LLM增强的贝叶斯优化演示
使用项目中已实现的llmbo模块

特性:
1. 使用LLM Warm Start生成初始点（而非随机）
2. 使用增强核函数（LLM-enhanced kernel）
3. 使用动态采样策略（Dynamic sampling）
4. random_state=None确保每次运行结果不同

注意：
- 此文件应放在BO/目录下（与BO_demo.py同级）
- 历史经验存储功能：当前prompt中包含领域知识，未来可扩展为保存历史运行结果
"""

from llmbo import LLMBOOptimizer
from SPM import SPM
import numpy as np


def charging_time_compute(current1, charging_number, current2):
    """
    两阶段充电目标函数
    返回负的充电步数（BO求最大值，所以用负数）
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


if __name__ == "__main__":
    print("="*70)
    print("LLMBO Demo - LLM增强贝叶斯优化")
    print("="*70)
    
    # 参数边界定义
    pbounds = {
        "current1": (3, 6),           # 第一阶段充电电流 (A)
        "charging_number": (5, 25),   # 第一阶段持续步数
        "current2": (1, 4)            # 第二阶段充电电流 (A)
    }
    
    print("\n问题设置:")
    print(f"  优化目标: 最小化充电步数")
    print(f"  参数空间: {pbounds}")
    
    # 创建LLMBO优化器
    # 关键: random_state=None 使每次运行都不同
    print("\n初始化LLMBO优化器...")
    print("  策略1: LLM Warm Start (智能初始点)")
    print("  策略2: Enhanced Kernel (增强核函数)")  
    print("  策略3: Dynamic Sampling (动态采样)")
    print("  random_state: None (真随机，避免重复)")
    
    llmbo = LLMBOOptimizer(
        f=charging_time_compute,
        pbounds=pbounds,
        llm_model="gpt-3.5-turbo",
        random_state=None,              # 关键：None表示真随机
        use_enhanced_kernel=True,       # 启用增强核函数
        use_dynamic_sampling=True       # 启用动态采样
    )
    
    # 运行优化
    print("\n" + "="*70)
    print("开始优化")
    print("="*70)
    
    result = llmbo.maximize(
        init_points=5,                  # LLM生成5个智能初始点
        n_iter=30,                      # 贝叶斯优化30次迭代
        use_llm_warm_start=True         # 启用LLM Warm Start
    )
    
    # 显示最优结果
    print("\n" + "="*70)
    print("优化完成 - 最优结果")
    print("="*70)
    print(f"\n最优充电步数: {-result['target']:.0f} 步")
    print(f"最优参数:")
    print(f"  current1 = {result['params']['current1']:.4f} A")
    print(f"  charging_number = {result['params']['charging_number']:.2f} 步")
    print(f"  current2 = {result['params']['current2']:.4f} A")
    
    print("\n" + "="*70)
    print("运行完成")
    print("="*70)
    
    # 注意：历史经验存储功能
    # 当前版本: prompt中包含领域知识和物理约束
    # 未来改进: 可以将每次运行的最优参数存储到文件
    #          在下次运行时作为额外的经验注入到prompt中
    #          参考论文Figure 2中的"optimization history"部分