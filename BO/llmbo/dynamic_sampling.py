"""
策略3: LLM动态采样策略 (修复版)
每5次迭代让LLM分析历史数据，调整采样策略

修复内容:
1. 添加空列表检查，防止max()错误
2. 添加数据量检查，确保有足够数据才分析
3. 增强错误处理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils import LLMClient
from typing import List, Dict, Tuple
import numpy as np


class DynamicSamplingStrategy:
    """
    LLM动态采样策略生成器
    
    每5次迭代分析一次历史数据，让LLM建议下一步探索方向
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        """
        初始化动态采样策略
        
        参数:
            llm_model: LLM模型名称
        """
        self.llm_client = LLMClient(model_name=llm_model)
        self.analysis_interval = 5
        self.min_data_points = 3  # 最少需要3个数据点才分析
    
    def should_analyze(self, iteration: int) -> bool:
        """
        判断是否应该进行分析
        
        参数:
            iteration: 当前迭代次数（从0开始）
            
        返回:
            是否应该分析
        """
        return (iteration >= self.min_data_points) and ((iteration + 1) % self.analysis_interval == 0)
    
    def create_analysis_prompt(
        self,
        historical_results: List[Dict],
        pbounds: Dict[str, Tuple],
        current_iteration: int
    ) -> str:
        """
        创建LLM分析提示词
        
        参数:
            historical_results: 历史评估结果
            pbounds: 参数边界
            current_iteration: 当前迭代次数
            
        返回:
            Prompt字符串
        """
        # 整理历史数据
        recent_results = historical_results[-5:] if len(historical_results) >= 5 else historical_results
        best_result = max(historical_results, key=lambda x: x['target'])
        
        # 构建表格
        results_table = f"Last {len(recent_results)} evaluations:\n"
        results_table += "| # | current1 | charging_number | current2 | Steps |\n"
        results_table += "|---|----------|-----------------|----------|-------|\n"
        
        for i, res in enumerate(recent_results, 1):
            current1 = res['params']['current1']
            charging_number = res['params']['charging_number']
            current2 = res['params']['current2']
            steps = int(-res['target'])
            results_table += f"| {i} | {current1:.2f} | {charging_number:.2f} | {current2:.2f} | {steps} |\n"
        
        prompt = f"""
Analyze the battery charging optimization data and suggest the next exploration strategy.

CURRENT STATUS:
- Iteration: {current_iteration}
- Best result so far: {int(-best_result['target'])} steps
- Best parameters:
  * current1 = {best_result['params']['current1']:.4f}
  * charging_number = {best_result['params']['charging_number']:.4f}
  * current2 = {best_result['params']['current2']:.4f}

HISTORICAL DATA:
{results_table}

PARAMETER BOUNDS:
- current1: [{pbounds['current1'][0]}, {pbounds['current1'][1]}]
- charging_number: [{pbounds['charging_number'][0]}, {pbounds['charging_number'][1]}]
- current2: [{pbounds['current2'][0]}, {pbounds['current2'][1]}]

ANALYSIS TASK:
1. Identify patterns in the data (which parameters work best together?)
2. Identify under-explored regions (which parameter ranges haven't been tested much?)
3. Predict the next promising region to sample
4. Suggest a sampling strategy for the next 5 iterations

OUTPUT FORMAT:
Provide your analysis in the following JSON structure:
{{
  "patterns": "describe observed patterns",
  "unexplored_regions": "describe under-explored regions",
  "next_sampling_strategy": "describe strategy (EXPLOIT/EXPLORE/BALANCED)",
  "recommended_focus": {{
    "current1_range": [min, max],
    "charging_number_range": [min, max],
    "current2_range": [min, max]
  }},
  "reasoning": "explain your reasoning"
}}
"""
        return prompt
    
    def get_analysis(
        self,
        historical_results: List[Dict],
        pbounds: Dict[str, Tuple],
        current_iteration: int
    ) -> Dict:
        """
        获取LLM的采样策略分析
        
        参数:
            historical_results: 历史评估结果
            pbounds: 参数边界
            current_iteration: 当前迭代次数
            
        返回:
            分析结果字典
        """
        print(f"\n[动态采样分析] 第{current_iteration}次迭代 - 查询LLM...")
        
        # 修复1: 检查空列表
        if not historical_results or len(historical_results) == 0:
            print(f"  警告: 历史结果为空，返回默认策略")
            return {
                'next_sampling_strategy': 'BALANCED',
                'recommended_focus': pbounds,
                'reasoning': '历史数据为空，使用默认平衡策略'
            }
        
        # 修复2: 检查数据量
        if len(historical_results) < self.min_data_points:
            print(f"  警告: 历史结果过少({len(historical_results)}个)，返回探索策略")
            return {
                'next_sampling_strategy': 'EXPLORE',
                'recommended_focus': pbounds,
                'reasoning': f'数据点不足{self.min_data_points}个，优先探索'
            }
        
        # 修复3: 添加异常处理
        try:
            prompt = self.create_analysis_prompt(
                historical_results=historical_results,
                pbounds=pbounds,
                current_iteration=current_iteration
            )
            
            response = self.llm_client.query(
                prompt=prompt,
                return_json=True
            )
            
            # 解析JSON
            import json
            analysis = json.loads(response)
            print(f"[动态采样分析] 建议策略: {analysis['next_sampling_strategy']}")
            return analysis
            
        except Exception as e:
            print(f"  警告: LLM分析失败 - {str(e)}")
            # 返回默认分析
            return {
                'next_sampling_strategy': 'BALANCED',
                'recommended_focus': pbounds,
                'reasoning': f'LLM分析失败: {str(e)}，使用默认策略'
            }


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试LLM动态采样策略 (修复版)")
    print("="*70)
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    strategy = DynamicSamplingStrategy()
    
    # 测试1: 空列表
    print("\n测试1: 空列表处理")
    print("-"*70)
    empty_results = []
    analysis = strategy.get_analysis(
        historical_results=empty_results,
        pbounds=pbounds,
        current_iteration=5
    )
    print(f"结果: {analysis['next_sampling_strategy']}")
    print(f"原因: {analysis['reasoning']}")
    assert analysis['next_sampling_strategy'] == 'BALANCED'
    print("测试1 通过")
    
    # 测试2: 数据不足
    print("\n测试2: 数据不足处理")
    print("-"*70)
    few_results = [
        {'params': {'current1': 4.5, 'charging_number': 12.0, 'current2': 2.0}, 'target': -95.0},
        {'params': {'current1': 5.2, 'charging_number': 8.0, 'current2': 1.8}, 'target': -88.0}
    ]
    analysis = strategy.get_analysis(
        historical_results=few_results,
        pbounds=pbounds,
        current_iteration=2
    )
    print(f"结果: {analysis['next_sampling_strategy']}")
    print(f"原因: {analysis['reasoning']}")
    assert analysis['next_sampling_strategy'] == 'EXPLORE'
    print("测试2 通过")
    
    # 测试3: 正常数据
    print("\n测试3: 正常数据处理")
    print("-"*70)
    normal_results = [
        {'params': {'current1': 4.5, 'charging_number': 12.0, 'current2': 2.0}, 'target': -95.0},
        {'params': {'current1': 5.2, 'charging_number': 8.0, 'current2': 1.8}, 'target': -88.0},
        {'params': {'current1': 3.8, 'charging_number': 18.0, 'current2': 2.2}, 'target': -102.0},
        {'params': {'current1': 4.9, 'charging_number': 14.0, 'current2': 2.5}, 'target': -87.0},
        {'params': {'current1': 4.2, 'charging_number': 16.0, 'current2': 1.9}, 'target': -91.0},
    ]
    
    print("尝试调用LLM进行分析...")
    analysis = strategy.get_analysis(
        historical_results=normal_results,
        pbounds=pbounds,
        current_iteration=5
    )
    print(f"结果: {analysis['next_sampling_strategy']}")
    print(f"原因: {analysis['reasoning']}")
    print("测试3 完成 (注意: 如果LLM调用失败会返回默认策略)")
    
    # 测试4: should_analyze逻辑
    print("\n测试4: should_analyze逻辑")
    print("-"*70)
    test_cases = [
        (0, False),
        (2, False),
        (4, True),
        (9, True),
        (10, False),
        (14, True)
    ]
    
    for iteration, expected in test_cases:
        result = strategy.should_analyze(iteration)
        status = "通过" if result == expected else "失败"
        print(f"  迭代 {iteration}: {result} (期望: {expected}) - {status}")
        assert result == expected
    
    print("\n" + "="*70)
    print("所有测试通过")
    print("="*70)