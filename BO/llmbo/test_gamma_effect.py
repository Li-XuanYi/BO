"""
测试：验证动态更新γ是否真的影响GP的预测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from llmbo.enhanced_kernel_v3 import LLMEnhancedKernelV3, get_llm_kernel_config_v3

print("="*70)
print("测试：动态γ是否影响GP预测")
print("="*70)

# 1. 准备一些训练数据
print("\n步骤1: 创建训练数据")
print("-"*70)
X_train = np.array([
    [4.0, 10.0, 2.0],
    [5.0, 15.0, 3.0],
    [3.5, 20.0, 1.5],
    [4.5, 12.0, 2.5],
    [5.5, 8.0, 3.5]
])
y_train = np.array([-50, -45, -60, -48, -42])
print(f"训练数据: {X_train.shape[0]} 个点")

# 2. 创建LLM增强核函数
print("\n步骤2: 创建LLM增强核函数")
print("-"*70)
config = get_llm_kernel_config_v3()
kernel = LLMEnhancedKernelV3(
    length_scales=config['length_scales'],
    coupling_matrix=config['coupling_matrix'],
    coupling_strength=0.3  # 初始γ=0.3
)
print(f"初始γ: {kernel.coupling_strength:.4f}")

# 3. 创建GP并拟合
print("\n步骤3: 创建GP并拟合")
print("-"*70)
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp.fit(X_train, y_train)
print("GP拟合完成")

# 4. 在某个测试点进行预测（γ=0.3时）
print("\n步骤4: 使用初始γ预测")
print("-"*70)
X_test = np.array([[4.5, 11.0, 2.8]])
mean_before, std_before = gp.predict(X_test, return_std=True)
print(f"测试点: {X_test[0]}")
print(f"预测均值: {mean_before[0]:.4f}")
print(f"预测标准差: {std_before[0]:.4f}")

# 5. 直接修改kernel的γ
print("\n步骤5: 修改γ（0.3 → 0.6）")
print("-"*70)
print(f"修改前 - kernel.coupling_strength: {kernel.coupling_strength:.4f}")
print(f"修改前 - gp.kernel_.coupling_strength: {gp.kernel_.coupling_strength:.4f}")

# 修改kernel实例
kernel.coupling_strength = 0.6

print(f"修改后 - kernel.coupling_strength: {kernel.coupling_strength:.4f}")
print(f"修改后 - gp.kernel_.coupling_strength: {gp.kernel_.coupling_strength:.4f}")

# 6. 不重新拟合，直接用同一个GP预测
print("\n步骤6: 不重新拟合，直接预测")
print("-"*70)
mean_after_no_refit, std_after_no_refit = gp.predict(X_test, return_std=True)
print(f"预测均值: {mean_after_no_refit[0]:.4f}")
print(f"预测标准差: {std_after_no_refit[0]:.4f}")

# 7. 重新拟合后预测
print("\n步骤7: 重新拟合GP后预测")
print("-"*70)
gp.fit(X_train, y_train)
mean_after_refit, std_after_refit = gp.predict(X_test, return_std=True)
print(f"预测均值: {mean_after_refit[0]:.4f}")
print(f"预测标准差: {std_after_refit[0]:.4f}")

# 8. 结果分析
print("\n" + "="*70)
print("结果分析")
print("="*70)

mean_change_no_refit = abs(mean_after_no_refit[0] - mean_before[0])
std_change_no_refit = abs(std_after_no_refit[0] - std_before[0])

mean_change_refit = abs(mean_after_refit[0] - mean_before[0])
std_change_refit = abs(std_after_refit[0] - std_before[0])

print(f"\n不重新拟合的情况:")
print(f"  均值变化: {mean_change_no_refit:.6f}")
print(f"  标准差变化: {std_change_no_refit:.6f}")
print(f"  结论: {'有明显变化' if mean_change_no_refit > 0.01 or std_change_no_refit > 0.01 else '几乎无变化'}")

print(f"\n重新拟合的情况:")
print(f"  均值变化: {mean_change_refit:.6f}")
print(f"  标准差变化: {std_change_refit:.6f}")
print(f"  结论: {'有明显变化' if mean_change_refit > 0.01 or std_change_refit > 0.01 else '几乎无变化'}")

print("\n" + "="*70)
print("最终结论")
print("="*70)

if mean_change_no_refit < 0.001 and std_change_no_refit < 0.001:
    print("❌ 问题：直接修改γ不会影响已拟合的GP预测")
    print("   原因：sklearn的GP在fit时会复制kernel，修改原kernel无效")
    print("   解决：需要在每次更新γ后重新拟合GP")
    print("\n建议修改：在_update_coupling_strength_internal中")
    print("   1. 更新kernel.coupling_strength")
    print("   2. 强制重新拟合GP: gp.fit(X, y)")
else:
    print("✓ 直接修改γ会影响GP预测（无需重新拟合）")

if mean_change_refit > 0.01 or std_change_refit > 0.01:
    print("\n✓ 重新拟合后，γ的变化会显著影响GP预测")
    print(f"   均值变化: {mean_change_refit:.4f}")
    print(f"   标准差变化: {std_change_refit:.4f}")
else:
    print("\n⚠️ 即使重新拟合，γ的变化也很小")
    print("   可能需要加大γ的调整幅度")

print("\n" + "="*70)