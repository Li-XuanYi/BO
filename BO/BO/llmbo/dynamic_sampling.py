"""
策略3: LLM动态采样策略
每5次迭代让LLM分析历史数据，调整采样策略
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
        self.analysis_interval = 5  # 每5次迭代分析一次
    
    def should_analyze(self, iteration: int) -> bool:
        """
        判断是否应该进行分析
        
        参数:
            iteration: 当前迭代次数（从0开始）
            
        返回:
            是否应该分析
        """
        return (iteration > 0) and ((iteration + 1) % self.analysis_interval == 0)
    
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
        recent_results = historical_results[-5:]  # 取最近5个结果
        best_result = max(historical_results, key=lambda x: x['target'])
        
        # 构建表格
        results_table = "Last 5 evaluations:\n"
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
        try:
            import json
            analysis = json.loads(response)
            print(f"[动态采样分析] 建议策略: {analysis['next_sampling_strategy']}")
            return analysis
        except:
            print(f"[动态采样分析] 警告: 无法解析LLM响应")
            # 返回默认分析
            return {
                'next_sampling_strategy': 'BALANCED',
                'recommended_focus': pbounds,
                'reasoning': '默认平衡策略'
            }


# 测试代码
if __name__ == "__main__":
    print("测试LLM动态采样策略")
    print("="*70)
    
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    # 模拟历史数据
    historical_results = [
        {'params': {'current1': 4.5, 'charging_number': 12.0, 'current2': 2.0}, 'target': -95.0},
        {'params': {'current1': 5.2, 'charging_number': 8.0, 'current2': 1.8}, 'target': -88.0},
        {'params': {'current1': 3.8, 'charging_number': 18.0, 'current2': 2.2}, 'target': -102.0},
        {'params': {'current1': 4.9, 'charging_number': 14.0, 'current2': 2.5}, 'target': -87.0},
        {'params': {'current1': 4.2, 'charging_number': 16.0, 'current2': 1.9}, 'target': -91.0},
    ]
    
    strategy = DynamicSamplingStrategy()
    
    print("\n测试是否需要分析:")
    for iter_num in [0, 4, 5, 9, 10]:
        should_analyze = strategy.should_analyze(iter_num)
        print(f"  迭代 {iter_num}: {should_analyze}")
    
    print("\n测试LLM分析（第5次迭代）:")
    print("-"*70)
    analysis = strategy.get_analysis(
        historical_results=historical_results,
        pbounds=pbounds,
        current_iteration=5
    )
    
    print("\n分析结果:")
    print(f"  策略: {analysis.get('next_sampling_strategy', 'N/A')}")
    print(f"  推荐焦点: {analysis.get('recommended_focus', 'N/A')}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)