"""
自适应Acquisition Function
为策略3（LLM动态采样）做准备
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_opt.acquisition import UpperConfidenceBound
import numpy as np


class AdaptiveAcquisition(UpperConfidenceBound):
    """
    自适应的UCB Acquisition Function
    
    功能：
    1. 动态调整kappa参数
    2. 记录迭代历史
    3. 为LLM分析预留接口
    """
    
    def __init__(self, kappa=2.5, kappa_decay=1.0, random_state=None):
        """
        初始化
        
        参数:
            kappa: 初始探索参数
            kappa_decay: kappa衰减系数（每次迭代后 kappa *= kappa_decay）
            random_state: 随机种子
        """
        # 注意：UpperConfidenceBound 使用 exploration_decay 而不是 kappa_decay
        super().__init__(
            kappa=kappa, 
            exploration_decay=kappa_decay,  # 修正：使用正确的参数名
            exploration_decay_delay=0,      # 修正：使用正确的参数名
            random_state=random_state
        )
        
        # 记录历史
        self.iteration = 0
        self.kappa_history = []
        self.suggested_points = []
        
        # 初始kappa
        self.initial_kappa = kappa
        self.current_kappa = kappa
        self.kappa_decay = kappa_decay  # 保存衰减系数
        
        print(f"AdaptiveAcquisition初始化完成")
        print(f"  初始kappa: {kappa}")
        print(f"  kappa衰减: {kappa_decay}")
    
    def update_kappa(self, new_kappa):
        """
        手动更新kappa值（供LLM调整使用）
        
        参数:
            new_kappa: 新的kappa值
        """
        old_kappa = self.current_kappa
        self.current_kappa = new_kappa
        self.kappa = new_kappa
        
        print(f"  [迭代{self.iteration}] kappa调整: {old_kappa:.3f} -> {new_kappa:.3f}")
    
    def adjust_exploration(self, mode='auto', record_history=True):
        """
        调整探索策略
        
        参数:
            mode: 'explore' (增加探索), 'exploit' (增加利用), 'auto' (自动)
            record_history: 是否记录历史（测试时设为True）
        """
        # 如果是测试模式，记录历史
        if record_history and self.iteration == 0:
            self.iteration += 1
            self.kappa_history.append(self.current_kappa)
        
        if mode == 'explore':
            # 增加探索：kappa变大
            new_kappa = min(self.current_kappa * 1.5, 5.0)
            self.update_kappa(new_kappa)
        elif mode == 'exploit':
            # 增加利用：kappa变小
            new_kappa = max(self.current_kappa * 0.7, 0.5)
            self.update_kappa(new_kappa)
        elif mode == 'auto':
            # 自动衰减
            new_kappa = self.current_kappa * self.kappa_decay
            self.update_kappa(new_kappa)
        
        # 记录调整后的kappa
        if record_history:
            self.kappa_history.append(self.current_kappa)
    
    def suggest(self, gp, target_space, n_random=10000, n_l_bfgs_b=10, fit_gp=True):
        """
        建议下一个采样点（重写父类方法）
        
        参数:
            gp: 高斯过程模型
            target_space: 目标空间
            n_random: 随机采样点数
            n_l_bfgs_b: L-BFGS-B优化起始点数
            fit_gp: 是否拟合GP
            
        返回:
            建议的采样点
        """
        # 记录当前迭代
        self.iteration += 1
        self.kappa_history.append(self.current_kappa)
        
        # 调用父类的suggest方法
        suggestion = super().suggest(gp, target_space, n_random, n_l_bfgs_b, fit_gp)
        
        # 记录建议点
        self.suggested_points.append(suggestion)
        
        return suggestion
    
    def get_stats(self):
        """
        获取统计信息
        
        返回:
            统计信息字典
        """
        return {
            'total_iterations': self.iteration,
            'current_kappa': self.current_kappa,
            'initial_kappa': self.initial_kappa,
            'kappa_history': self.kappa_history,
            'average_kappa': np.mean(self.kappa_history) if self.kappa_history else 0
        }
    
    def reset(self):
        """
        重置为初始状态
        """
        self.iteration = 0
        self.current_kappa = self.initial_kappa
        self.kappa = self.initial_kappa
        self.kappa_history = []
        self.suggested_points = []


# 测试代码
if __name__ == "__main__":
    print("测试AdaptiveAcquisition")
    print("="*70)
    
    # 创建实例
    acq = AdaptiveAcquisition(kappa=2.5, kappa_decay=0.95)
    
    print("\n模拟10次迭代：")
    print("-"*70)
    
    # 模拟10次迭代
    for i in range(10):
        print(f"\n迭代 {i+1}:")
        print(f"  当前kappa: {acq.current_kappa:.3f}")
        
        # 模拟不同的调整策略
        if i == 3:
            print("  检测到收敛缓慢，增加探索...")
            acq.adjust_exploration(mode='explore')
        elif i == 7:
            print("  检测到找到好区域，增加利用...")
            acq.adjust_exploration(mode='exploit')
        else:
            print("  使用自动衰减...")
            acq.adjust_exploration(mode='auto')
    
    # 显示统计
    print("\n" + "="*70)
    print("统计信息：")
    print("-"*70)
    stats = acq.get_stats()
    print(f"总迭代次数: {stats['total_iterations']}")
    print(f"初始kappa: {stats['initial_kappa']:.3f}")
    print(f"最终kappa: {stats['current_kappa']:.3f}")
    print(f"平均kappa: {stats['average_kappa']:.3f}")
    print(f"kappa历史: {[f'{k:.3f}' for k in stats['kappa_history']]}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)