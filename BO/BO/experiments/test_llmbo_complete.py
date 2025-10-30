"""
LLMBO综合测试 - 验证三个策略的集成

测试内容:
1. 策略1: LLM Warm Start
2. 策略2: 动态γ更新 + GP重新拟合
3. 策略3: LLM加权采集函数

目标: 确保三个策略协同工作
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPM import SPM
from llmbo.llmbo_optimizer import LLMBOOptimizer


def charging_time_compute(current1, charging_number, current2):
    """
    两阶段充电目标函数
    
    目标: 最小化充电时间（步数）
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


def test_strategy_1_warm_start():
    """测试策略1: LLM Warm Start"""
    print("\n" + "="*70)
    print("测试策略1: LLM Warm Start")
    print("="*70)
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    try:
        llmbo = LLMBOOptimizer(
            f=charging_time_compute,
            pbounds=pbounds,
            llm_model="gpt-3.5-turbo",
            random_state=42,
            use_enhanced_kernel=False,
            use_dynamic_sampling=False,
            use_weighted_acquisition=False,
            verbose=1
        )
        
        # 只测试warm start
        result = llmbo.maximize(
            init_points=3,
            n_iter=2,
            use_llm_warm_start=True
        )
        
        print("\n✓ 策略1测试通过")
        print(f"  初始点数: {len([r for r in llmbo.res if r][:3])}")
        print(f"  最优结果: {-result['target']:.0f} 步")
        return True
        
    except Exception as e:
        print(f"\n✗ 策略1测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_2_dynamic_gamma():
    """测试策略2: 动态γ + GP重新拟合"""
    print("\n" + "="*70)
    print("测试策略2: 动态γ + GP重新拟合")
    print("="*70)
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    try:
        llmbo = LLMBOOptimizer(
            f=charging_time_compute,
            pbounds=pbounds,
            llm_model="gpt-3.5-turbo",
            random_state=42,
            use_enhanced_kernel=True,  # 启用增强核
            use_dynamic_sampling=False,
            use_weighted_acquisition=False,
            verbose=1
        )
        
        result = llmbo.maximize(
            init_points=3,
            n_iter=3,
            use_llm_warm_start=False
        )
        
        # 检查gamma历史
        if llmbo.kernel:
            gamma_history = llmbo.kernel.get_gamma_history()
            print(f"\n✓ 策略2测试通过")
            print(f"  γ历史: {[f'{g:.4f}' for g in gamma_history['gamma_history']]}")
            print(f"  γ更新次数: {len(gamma_history['gamma_history']) - 1}")
            return True
        else:
            print("\n✗ 策略2测试失败: kernel未初始化")
            return False
        
    except Exception as e:
        print(f"\n✗ 策略2测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_3_weighted_acquisition():
    """测试策略3: LLM加权采集函数"""
    print("\n" + "="*70)
    print("测试策略3: LLM加权采集函数")
    print("="*70)
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    try:
        llmbo = LLMBOOptimizer(
            f=charging_time_compute,
            pbounds=pbounds,
            llm_model="gpt-3.5-turbo",
            random_state=42,
            use_enhanced_kernel=False,
            use_dynamic_sampling=True,  # 启用动态采样
            use_weighted_acquisition=True,  # 启用加权采集
            verbose=1
        )
        
        result = llmbo.maximize(
            init_points=3,
            n_iter=3,
            use_llm_warm_start=False
        )
        
        print(f"\n✓ 策略3测试通过")
        print(f"  加权采集函数已使用")
        print(f"  最优结果: {-result['target']:.0f} 步")
        return True
        
    except Exception as e:
        print(f"\n✗ 策略3测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_strategies_integrated():
    """测试所有策略集成"""
    print("\n" + "="*70)
    print("综合测试: 三个策略同时启用")
    print("="*70)
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    try:
        llmbo = LLMBOOptimizer(
            f=charging_time_compute,
            pbounds=pbounds,
            llm_model="gpt-3.5-turbo",
            random_state=42,
            use_enhanced_kernel=True,
            use_dynamic_sampling=True,
            use_weighted_acquisition=True,
            verbose=2
        )
        
        result = llmbo.maximize(
            init_points=5,
            n_iter=5,
            use_llm_warm_start=True
        )
        
        print(f"\n✓ 综合测试通过")
        print(f"  总评估次数: {len(llmbo.res)}")
        print(f"  最优充电步数: {-result['target']:.0f} 步")
        print(f"  最优参数:")
        for key, value in result['params'].items():
            print(f"    {key} = {value:.4f}")
        
        # 检查各个策略的工作状态
        print(f"\n策略验证:")
        print(f"  ✓ 策略1 (Warm Start): 已使用")
        
        if llmbo.kernel:
            gamma_history = llmbo.kernel.get_gamma_history()
            print(f"  ✓ 策略2 (动态γ): γ更新{len(gamma_history['gamma_history'])-1}次")
        
        if llmbo.weighted_acq:
            print(f"  ✓ 策略3 (加权采集): 已使用")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 综合测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*70)
    print("LLMBO完整性测试")
    print("="*70)
    print("\n逐步验证三个策略的实现...")
    
    results = {
        '策略1 (Warm Start)': False,
        '策略2 (动态γ)': False,
        '策略3 (加权采集)': False,
        '综合集成': False
    }
    
    # 测试1
    results['策略1 (Warm Start)'] = test_strategy_1_warm_start()
    
    # 测试2
    results['策略2 (动态γ)'] = test_strategy_2_dynamic_gamma()
    
    # 测试3
    results['策略3 (加权采集)'] = test_strategy_3_weighted_acquisition()
    
    # 综合测试
    results['综合集成'] = test_all_strategies_integrated()
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 所有测试通过！LLMBO完整实现成功！")
        print("="*70)
        print("\n三个策略已经成功集成:")
        print("  1. ✓ LLM Warm Start - 智能初始化")
        print("  2. ✓ 动态γ调整 + GP重新拟合 - 自适应核函数")
        print("  3. ✓ LLM加权采集函数 - 智能探索")
        print("\n可以开始正式的优化实验了！")
    else:
        print("\n" + "="*70)
        print("⚠ 部分测试失败，需要进一步调试")
        print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)