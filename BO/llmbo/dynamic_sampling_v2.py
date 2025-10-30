"""
策略3: LLM增强的候选采样 V2 - 核心创新！

这是LLMBO论文的核心创新点！
论文方程8: α_EI^LLM(θ) = E[max(f_min - f(θ), 0)] · W_LLM(θ | D)
论文方程9: W_LLM = Π (1/√(2πσ_j²)) · exp(-(θ_j - μ_j)²/(2σ_j²))

核心思想:
- LLM分析优化历史，识别敏感参数
- 动态调整采样权重，聚焦高潜力区域
- 平衡exploration（全局搜索）和exploitation（局部优化）

关键改进:
1. 实现论文的权重函数W_LLM
2. LLM动态调整μ和σ参数
3. 智能采样策略切换（EXPLORATION/EXPLOITATION/BALANCED）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils import LLMClient
from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import norm
from bayes_opt import UtilityFunction


class LLMEnhancedSamplingV2:
    """
    LLM增强的候选采样策略V2 - 实现论文方程8-9
    
    核心功能:
    1. LLM分析优化历史
    2. 动态调整采样权重W_LLM
    3. 智能选择采样策略
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        analysis_interval: int = 3,
        min_data_points: int = 3
    ):
        """
        初始化LLM增强采样策略
        
        参数:
            llm_model: LLM模型名称
            analysis_interval: LLM分析间隔（每N次迭代分析一次）
            min_data_points: 最少需要的数据点数量
        """
        self.llm_client = LLMClient(model_name=llm_model)
        self.analysis_interval = analysis_interval
        self.min_data_points = min_data_points
        
        # 权重函数参数（论文方程9）
        self.mu = {}  # 每个参数的均值μ
        self.sigma = {}  # 每个参数的标准差σ
        
        # 当前策略
        self.current_strategy = 'BALANCED'  # EXPLORATION, EXPLOITATION, BALANCED
        
        print("[LLMEnhancedSamplingV2] 初始化完成")
        print(f"  LLM模型: {llm_model}")
        print(f"  分析间隔: 每{analysis_interval}次迭代")
    
    def should_analyze(self, iteration: int, n_data: int) -> bool:
        """判断是否应该进行LLM分析"""
        return (
            n_data >= self.min_data_points and 
            (iteration + 1) % self.analysis_interval == 0
        )
    
    def analyze_and_update_weights(
        self,
        historical_results: List[Dict],
        pbounds: Dict[str, Tuple],
        current_iteration: int,
        verbose: bool = True
    ) -> Dict:
        """
        LLM分析优化历史，更新权重参数μ和σ
        
        这是核心方法！实现论文中的LLM智能分析
        
        参数:
            historical_results: 历史优化结果
            pbounds: 参数边界
            current_iteration: 当前迭代次数
            verbose: 是否打印详细信息
        
        返回:
            LLM分析结果字典
        """
        if not self.should_analyze(current_iteration, len(historical_results)):
            return {'strategy': self.current_strategy, 'updated': False}
        
        if verbose:
            print(f"\n[LLM分析] 迭代 {current_iteration}, 历史数据: {len(historical_results)} 条")
        
        try:
            # 构建LLM prompt
            prompt = self._create_analysis_prompt(
                historical_results=historical_results,
                pbounds=pbounds,
                current_iteration=current_iteration
            )
            
            # 调用LLM
            response = self.llm_client.query(prompt=prompt, return_json=True)
            
            # 解析响应
            import json
            analysis = json.loads(response)
            
            # 更新策略
            self.current_strategy = analysis.get('sampling_strategy', 'BALANCED')
            
            # 更新权重参数μ和σ（论文方程9）
            sensitive_params = analysis.get('sensitive_parameters', {})
            for param_name, param_info in sensitive_params.items():
                self.mu[param_name] = param_info.get('focus_mean', 0.0)
                self.sigma[param_name] = param_info.get('focus_std', 1.0)
            
            if verbose:
                print(f"  策略: {self.current_strategy}")
                print(f"  敏感参数: {list(sensitive_params.keys())}")
                print(f"  更新μ/σ: {len(self.mu)} 个参数")
                reasoning = analysis.get('reasoning', 'N/A')
                print(f"  LLM推理: {reasoning[:100]}...")
            
            analysis['updated'] = True
            return analysis
            
        except Exception as e:
            print(f"  警告: LLM分析失败 - {e}")
            return {'strategy': self.current_strategy, 'updated': False, 'error': str(e)}
    
    def compute_llm_weight(
        self,
        theta: np.ndarray,
        param_names: List[str],
        pbounds: Dict[str, Tuple]
    ) -> float:
        """
        计算LLM权重W_LLM（论文方程9）
        
        W_LLM = Π (1/√(2πσ_j²)) · exp(-(θ_j - μ_j)²/(2σ_j²))
        
        参数:
            theta: 候选参数向量
            param_names: 参数名称列表
            pbounds: 参数边界
        
        返回:
            权重值（越高表示越值得采样）
        """
        if len(self.mu) == 0 or len(self.sigma) == 0:
            # 如果还没有LLM分析，返回均匀权重
            return 1.0
        
        weight = 1.0
        
        for i, param_name in enumerate(param_names):
            if param_name in self.mu and param_name in self.sigma:
                mu_j = self.mu[param_name]
                sigma_j = self.sigma[param_name]
                theta_j = theta[i]
                
                # 计算高斯权重
                # (1/√(2πσ²)) · exp(-(θ - μ)²/(2σ²))
                gauss_weight = norm.pdf(theta_j, loc=mu_j, scale=sigma_j)
                weight *= gauss_weight
            else:
                # 未指定的参数使用均匀权重
                weight *= 1.0
        
        return weight
    
    def get_enhanced_utility_function(
        self,
        base_utility: UtilityFunction,
        param_names: List[str],
        pbounds: Dict[str, Tuple]
    ):
        """
        获取LLM增强的效用函数（论文方程8）
        
        α_EI^LLM(θ) = E[max(f_min - f(θ), 0)] · W_LLM(θ | D)
        
        参数:
            base_utility: 基础效用函数（如EI）
            param_names: 参数名称列表
            pbounds: 参数边界
        
        返回:
            增强的效用函数
        """
        def enhanced_utility(x, gp, y_max):
            # 计算基础效用值（如EI）
            base_value = base_utility.utility(x, gp, y_max)
            
            # 计算LLM权重
            llm_weight = self.compute_llm_weight(x, param_names, pbounds)
            
            # 复合效用 = 基础效用 × LLM权重
            enhanced_value = base_value * llm_weight
            
            return enhanced_value
        
        return enhanced_utility
    
    def suggest_next_sample_region(
        self,
        historical_results: List[Dict],
        pbounds: Dict[str, Tuple]
    ) -> Dict:
        """
        根据当前策略建议下一个采样区域
        
        参数:
            historical_results: 历史优化结果
            pbounds: 参数边界
        
        返回:
            采样区域建议
        """
        if len(historical_results) == 0:
            return {'strategy': 'BALANCED', 'focus_region': pbounds}
        
        # 根据策略调整采样区域
        if self.current_strategy == 'EXPLOITATION':
            # 开发模式：聚焦最优点附近
            best_result = max(historical_results, key=lambda x: x['target'])
            best_params = best_result['params']
            
            # 在最优点附近创建缩小的采样区域
            focus_region = {}
            for param_name, (lower, upper) in pbounds.items():
                best_value = best_params[param_name]
                range_size = (upper - lower) * 0.2  # 缩小到20%范围
                focus_lower = max(lower, best_value - range_size / 2)
                focus_upper = min(upper, best_value + range_size / 2)
                focus_region[param_name] = (focus_lower, focus_upper)
            
            return {'strategy': 'EXPLOITATION', 'focus_region': focus_region}
        
        elif self.current_strategy == 'EXPLORATION':
            # 探索模式：扩大采样范围
            return {'strategy': 'EXPLORATION', 'focus_region': pbounds}
        
        else:  # BALANCED
            # 平衡模式：正常采样
            return {'strategy': 'BALANCED', 'focus_region': pbounds}
    
    def _create_analysis_prompt(
        self,
        historical_results: List[Dict],
        pbounds: Dict[str, Tuple],
        current_iteration: int
    ) -> str:
        """创建LLM分析的prompt"""
        # 获取最近结果和最佳结果
        recent_results = historical_results[-5:] if len(historical_results) >= 5 else historical_results
        best_result = max(historical_results, key=lambda x: x['target'])
        
        # 构建结果表格
        results_table = f"最近 {len(recent_results)} 次评估:\n"
        results_table += "| # | current1 | charging_number | current2 | 充电步数 |\n"
        results_table += "|---|----------|-----------------|----------|----------|\n"
        
        for i, res in enumerate(recent_results, 1):
            current1 = res['params']['current1']
            charging_number = res['params']['charging_number']
            current2 = res['params']['current2']
            steps = int(-res['target'])
            results_table += f"| {i} | {current1:.2f} | {charging_number:.2f} | {current2:.2f} | {steps} |\n"
        
        # 构建prompt
        prompt = f"""
你是电池充电优化专家。分析历史数据并给出智能采样策略建议。

当前状态:
- 迭代次数: {current_iteration}
- 历史数据量: {len(historical_results)}
- 当前最优: {int(-best_result['target'])} 步
- 最优参数:
  * current1 = {best_result['params']['current1']:.4f}
  * charging_number = {best_result['params']['charging_number']:.4f}
  * current2 = {best_result['params']['current2']:.4f}

{results_table}

参数边界:
- current1: [{pbounds['current1'][0]}, {pbounds['current1'][1]}] (第一阶段电流, A)
- charging_number: [{pbounds['charging_number'][0]}, {pbounds['charging_number'][1]}] (第一阶段步数)
- current2: [{pbounds['current2'][0]}, {pbounds['current2'][1]}] (第二阶段电流, A)

任务: 分析优化进展，给出下一步采样策略

返回JSON格式:
{{
    "sampling_strategy": "EXPLORATION/EXPLOITATION/BALANCED",
    "sensitive_parameters": {{
        "current1": {{"focus_mean": 4.5, "focus_std": 0.3, "importance": "high"}},
        "charging_number": {{"focus_mean": 12.0, "focus_std": 2.0, "importance": "medium"}},
        "current2": {{"focus_mean": 2.5, "focus_std": 0.4, "importance": "high"}}
    }},
    "reasoning": "分析推理过程"
}}

策略说明:
- EXPLORATION: 优化刚开始或陷入局部最优，需要全局搜索
- EXPLOITATION: 找到好的区域，需要精细优化
- BALANCED: 正常平衡探索和开发

敏感参数说明:
- focus_mean: 下一步采样应聚焦的参数均值
- focus_std: 采样的标准差（越小越聚焦）
- importance: 参数重要性（high/medium/low）
"""
        return prompt
    
    def get_statistics(self) -> Dict:
        """获取采样统计信息"""
        return {
            'current_strategy': self.current_strategy,
            'n_parameters_tracked': len(self.mu),
            'mu_values': self.mu.copy(),
            'sigma_values': self.sigma.copy()
        }


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试LLM增强采样策略V2 - 论文方程8-9实现")
    print("="*70)
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    sampler = LLMEnhancedSamplingV2()
    
    # 模拟历史数据
    fake_history = [
        {'params': {'current1': 4.0, 'charging_number': 10.0, 'current2': 2.0}, 'target': -100.0},
        {'params': {'current1': 4.5, 'charging_number': 15.0, 'current2': 2.5}, 'target': -95.0},
        {'params': {'current1': 5.0, 'charging_number': 12.0, 'current2': 2.2}, 'target': -90.0},
        {'params': {'current1': 4.7, 'charging_number': 13.0, 'current2': 2.3}, 'target': -88.0},
    ]
    
    print("\n测试1: LLM分析和权重更新")
    print("-"*70)
    analysis = sampler.analyze_and_update_weights(
        historical_results=fake_history,
        pbounds=pbounds,
        current_iteration=3,
        verbose=True
    )
    
    print("\n测试2: 计算LLM权重（方程9）")
    print("-"*70)
    theta = np.array([4.5, 12.0, 2.5])
    param_names = ['current1', 'charging_number', 'current2']
    
    weight = sampler.compute_llm_weight(theta, param_names, pbounds)
    print(f"候选点 {theta} 的LLM权重: {weight:.6f}")
    
    print("\n测试3: 采样区域建议")
    print("-"*70)
    region_advice = sampler.suggest_next_sample_region(fake_history, pbounds)
    print(f"策略: {region_advice['strategy']}")
    print(f"聚焦区域: {region_advice['focus_region']}")
    
    print("\n测试4: 统计信息")
    print("-"*70)
    stats = sampler.get_statistics()
    print(f"当前策略: {stats['current_strategy']}")
    print(f"跟踪参数数: {stats['n_parameters_tracked']}")
    
    print("\n" + "="*70)
    print("LLM增强采样策略V2测试完成")
    print("="*70)
