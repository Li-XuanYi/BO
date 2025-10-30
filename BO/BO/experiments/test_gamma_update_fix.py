"""
测试：验证γ更新修复是否生效

目标：确认修改γ后重新拟合GP能使预测发生变化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from llmbo.enhanced_kernel import LLMEnhancedKernel, get_llm_kernel_config


def test_gamma_update_with_refit():
    """
    测试修复后的流程：
    1. 创建GP并拟合
    2. 预测（使用初始γ）
    3. 修改γ
    4. 重新拟合GP
    5. 预测（使用新的γ）
    6. 验证预测是否发生变化
    """
    print("="*70)
    print("测试：γ更新 + GP重新拟合")
    print("="*70)
    
    # 步骤1: 创建训练数据
    print("\n步骤1: 创建训练数据")
    print("-"*70)
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, size=(5, 3))
    y_train = np.sum(X_train**2, axis=1) + np.random.randn(5) * 0.1
    print(f"训练数据: {len(X_train)} 个点")
    
    # 步骤2: 创建增强核函数
    print("\n步骤2: 创建LLM增强核函数")
    print("-"*70)
    config = get_llm_kernel_config()
    kernel = LLMEnhancedKernel(
        length_scales=config['length_scales'],
        coupling_matrix=config['coupling_matrix'],
        coupling_strength=0.3
    )
    print(f"初始γ: {kernel.coupling_strength:.4f}")
    
    # 步骤3: 创建GP并拟合
    print("\n步骤3: 创建GP并拟合")
    print("-"*70)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=0)
    gp.fit(X_train, y_train)
    print("GP拟合完成")
    
    # 步骤4: 使用初始γ预测
    print("\n步骤4: 使用初始γ预测")
    print("-"*70)
    X_test = np.array([[4.5, 11.0, 2.8]])
    mean_before, std_before = gp.predict(X_test, return_std=True)
    print(f"测试点: {X_test[0]}")
    print(f"预测均值: {mean_before[0]:.4f}")
    print(f"预测标准差: {std_before[0]:.4f}")
    
    # 步骤5: 修改γ（模拟优化过程中的更新）
    print("\n步骤5: 修改γ（0.3 → 0.6）")
    print("-"*70)
    print(f"修改前 - kernel.coupling_strength: {kernel.coupling_strength:.4f}")
    print(f"修改前 - gp.kernel_.coupling_strength: {gp.kernel_.coupling_strength:.4f}")
    
    # 模拟update_gamma（直接修改）
    kernel.coupling_strength = 0.6
    
    print(f"修改后 - kernel.coupling_strength: {kernel.coupling_strength:.4f}")
    print(f"修改后 - gp.kernel_.coupling_strength: {gp.kernel_.coupling_strength:.4f}")
    print("  注意: gp.kernel_仍然是旧的，因为sklearn克隆了kernel")
    
    # 步骤6: 不重新拟合，直接预测（应该没有变化）
    print("\n步骤6: 不重新拟合，直接预测")
    print("-"*70)
    mean_without_refit, std_without_refit = gp.predict(X_test, return_std=True)
    print(f"预测均值: {mean_without_refit[0]:.4f}")
    print(f"预测标准差: {std_without_refit[0]:.4f}")
    
    # 步骤7: 【修复方案】重新设置kernel并拟合
    print("\n步骤7: 【修复方案】重新设置kernel并拟合")
    print("-"*70)
    gp.kernel = kernel  # 重新设置kernel
    gp.fit(X_train, y_train)  # 重新拟合
    print("GP已重新拟合")
    
    # 步骤8: 重新拟合后预测
    print("\n步骤8: 重新拟合后预测")
    print("-"*70)
    mean_after, std_after = gp.predict(X_test, return_std=True)
    print(f"预测均值: {mean_after[0]:.4f}")
    print(f"预测标准差: {std_after[0]:.4f}")
    
    # 步骤9: 分析结果
    print("\n" + "="*70)
    print("结果分析")
    print("="*70)
    
    print("\n不重新拟合的情况:")
    print(f"  均值变化: {abs(mean_without_refit[0] - mean_before[0]):.6f}")
    print(f"  标准差变化: {abs(std_without_refit[0] - std_before[0]):.6f}")
    if abs(mean_without_refit[0] - mean_before[0]) < 1e-6:
        print(f"  结论: ❌ 几乎无变化（预期行为）")
    
    print("\n重新拟合的情况:")
    print(f"  均值变化: {abs(mean_after[0] - mean_before[0]):.6f}")
    print(f"  标准差变化: {abs(std_after[0] - std_before[0]):.6f}")
    if abs(mean_after[0] - mean_before[0]) > 0.1:
        print(f"  结论: ✓ 有明显变化（修复成功）")
    else:
        print(f"  结论: ⚠ 变化较小（可能需要检查）")
    
    # 步骤10: 最终结论
    print("\n" + "="*70)
    print("最终结论")
    print("="*70)
    
    if abs(mean_without_refit[0] - mean_before[0]) < 1e-6 and \
       abs(mean_after[0] - mean_before[0]) > 0.1:
        print("✓ 修复方案有效！")
        print("  - 直接修改γ不会影响已拟合的GP预测")
        print("  - 重新设置kernel + 重新拟合后，γ的变化显著影响预测")
        print("\n建议：在LLMBO的_update_coupling_strength_internal中")
        print("  1. 更新kernel.coupling_strength")
        print("  2. 更新gp.kernel = kernel")
        print("  3. 重新拟合: gp.fit(X_train, y_train)")
        return True
    else:
        print("❌ 测试未通过，需要进一步检查")
        return False


if __name__ == "__main__":
    success = test_gamma_update_with_refit()
    
    if success:
        print("\n" + "="*70)
        print("第一步完成：γ更新修复验证通过")
        print("="*70)
        print("\n可以进入第二步：实现策略3的加权采集函数")
    else:
        print("\n需要进一步调试")