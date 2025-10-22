"""
LLM智能Gamma调整模块
让LLM根据优化历史分析并建议gamma调整策略
参考论文Figure 3的Surrogate Modeling Prompt设计
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils import LLMClient
from typing import List, Dict
import numpy as np


class LLMGammaAdvisor:
    """
    LLM Gamma智能顾问
    
    分析优化历史,给出gamma调整建议
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.llm_client = LLMClient(model_name=llm_model)
        self.analysis_interval = 5
    
    def should_consult_llm(self, iteration: int) -> bool:
        """
        判断是否应该咨询LLM
        
        每5次迭代咨询一次
        """
        return (iteration > 0) and (iteration % self.analysis_interval == 0)
    
    def create_gamma_analysis_prompt(
        self,
        optimization_history: List[Dict],
        current_gamma: float,
        gamma_history: List[float],
        f_min_history: List[float]
    ) -> str:
        """
        创建Gamma分析Prompt (基于论文Figure 3设计)
        """
        
        recent_results = optimization_history[-10:] if len(optimization_history) >= 10 else optimization_history
        best_result = max(optimization_history, key=lambda x: x['target'])
        
        prompt = f"""As an expert in electrochemistry and Bayesian optimization, analyze the optimization progress for lithium-ion battery fast charging and provide guidance on adjusting the coupling strength (gamma) in the GP surrogate model.

BACKGROUND: Composite Kernel Structure
The GP kernel combines a base Matern kernel with an electrochemical coupling term:
k(theta, theta') = k_base(theta, theta') + gamma * k_coupling(theta, theta')

Where:
- k_base: Captures general smoothness
- k_coupling: Encodes parameter correlations (current1 <-> charging_number <-> current2)
- gamma: Coupling strength (controls how much we trust the physical coupling model)

CURRENT OPTIMIZATION STATUS:
- Total iterations: {len(optimization_history)}
- Current gamma: {current_gamma:.4f}
- Best charging time so far: {int(-best_result['target'])} steps
- Best parameters: current1={best_result['params']['current1']:.2f}A, charging_number={best_result['params']['charging_number']:.1f}, current2={best_result['params']['current2']:.2f}A

GAMMA EVOLUTION:
{self._format_gamma_history(gamma_history, f_min_history)}

RECENT OPTIMIZATION TRAJECTORY:
{self._format_recent_results(recent_results)}

CONVERGENCE ANALYSIS:
{self._analyze_convergence(f_min_history)}

TASK: Surrogate Model Tuning
Analyze the optimization behavior and recommend gamma adjustment:

1. CONVERGENCE PATTERN:
   - Is the optimization converging rapidly (consistent improvement)?
   - Is it stagnating (little improvement)?
   - Is it oscillating (improvement then deterioration)?

2. PARAMETER COUPLING INSIGHTS:
   - Are the best points showing strong correlation between parameters?
   - Example: High current1 consistently paired with short charging_number?
   - Or are good points scattered randomly in parameter space?

3. GAMMA RECOMMENDATION:
   Based on electrochemical principles:
   
   IF rapid convergence + clear parameter patterns:
   → INCREASE gamma (strengthen coupling)
   → Rationale: Physical relationships are helping, reinforce them
   
   IF stagnation + unclear patterns:
   → DECREASE gamma (weaken coupling)
   → Rationale: Coupling model may be too restrictive, allow more freedom
   
   IF oscillation:
   → MODERATE adjustment (small decrease)
   → Rationale: May be overfitting, need to balance

OUTPUT FORMAT (JSON):
{{
  "convergence_assessment": "rapid/moderate/slow/stagnating/oscillating",
  "coupling_evidence": "strong/moderate/weak - explain observed patterns",
  "recommended_gamma": <float between 0.01 and 1.0>,
  "adjustment_ratio": <recommended_gamma / current_gamma>,
  "confidence": "high/medium/low",
  "reasoning": "detailed explanation of recommendation",
  "expected_impact": "what should happen if gamma is adjusted as recommended"
}}

IMPORTANT:
- Consider both convergence speed AND parameter correlation patterns
- Be conservative: suggest small adjustments (ratio between 0.8 and 1.2) unless very confident
- High gamma (>0.5): Strong coupling, trust physical model
- Low gamma (<0.2): Weak coupling, more data-driven exploration
- Medium gamma (0.2-0.5): Balanced approach
"""
        return prompt
    
    def _format_gamma_history(self, gamma_history: List[float], f_min_history: List[float]) -> str:
        """格式化gamma演化历史"""
        if len(gamma_history) < 2 or len(f_min_history) == 0:
            return "Insufficient history (initial phase)"
        
        history_str = "Iteration | Gamma   | Best f_min | Change\n"
        history_str += "-" * 50 + "\n"
        
        for i in range(min(len(gamma_history), len(f_min_history))):
            gamma = gamma_history[i]
            f_min = f_min_history[i] if i < len(f_min_history) else 0
            
            if i == 0:
                change = "initial"
            else:
                gamma_change = ((gamma - gamma_history[i-1]) / gamma_history[i-1]) * 100
                change = f"{gamma_change:+.2f}%"
            
            history_str += f"{i:<9} | {gamma:<7.4f} | {f_min:<10.2f} | {change}\n"
        
        return history_str
    
    def _format_recent_results(self, results: List[Dict]) -> str:
        """格式化最近的优化结果"""
        if not results:
            return "No results yet"
        
        table = "Iter | Steps | current1 | chg_num | current2 | Notes\n"
        table += "-" * 60 + "\n"
        
        best_so_far = float('inf')
        for i, res in enumerate(results, 1):
            steps = int(-res['target'])
            c1 = res['params']['current1']
            cn = res['params']['charging_number']
            c2 = res['params']['current2']
            
            note = ""
            if steps < best_so_far:
                note = "NEW BEST"
                best_so_far = steps
            
            table += f"{i:<4} | {steps:<5} | {c1:<8.2f} | {cn:<7.1f} | {c2:<8.2f} | {note}\n"
        
        return table
    
    def _analyze_convergence(self, f_min_history: List[float]) -> str:
        """分析收敛趋势"""
        if len(f_min_history) < 3:
            return "Insufficient data for convergence analysis"
        
        recent = f_min_history[-5:]
        improvements = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        
        avg_improvement = np.mean(improvements)
        improvement_std = np.std(improvements)
        
        analysis = f"Recent trend (last {len(recent)} iterations):\n"
        analysis += f"- Average improvement: {avg_improvement:.4f}\n"
        analysis += f"- Improvement std dev: {improvement_std:.4f}\n"
        
        if avg_improvement > 0.5:
            analysis += "- Pattern: Rapid convergence\n"
        elif avg_improvement > 0.1:
            analysis += "- Pattern: Moderate progress\n"
        elif avg_improvement > -0.1:
            analysis += "- Pattern: Stagnation\n"
        else:
            analysis += "- Pattern: Deterioration/Oscillation\n"
        
        return analysis
    
    def get_gamma_recommendation(
        self,
        optimization_history: List[Dict],
        current_gamma: float,
        gamma_history: List[float],
        f_min_history: List[float]
    ) -> Dict:
        """
        获取LLM的gamma调整建议
        
        返回:
            {
                'recommended_gamma': float,
                'confidence': str,
                'reasoning': str,
                'expected_impact': str
            }
        """
        print(f"\n[LLM Gamma Advisor] 分析优化状态...")
        
        if len(optimization_history) < 3:
            print("  数据不足,使用默认公式")
            return {
                'recommended_gamma': current_gamma,
                'confidence': 'low',
                'reasoning': '数据不足,保持当前gamma',
                'expected_impact': 'Continue initial exploration'
            }
        
        try:
            prompt = self.create_gamma_analysis_prompt(
                optimization_history=optimization_history,
                current_gamma=current_gamma,
                gamma_history=gamma_history,
                f_min_history=f_min_history
            )
            
            response = self.llm_client.query(
                prompt=prompt,
                return_json=True
            )
            
            import json
            analysis = json.loads(response)
            
            print(f"[LLM Gamma Advisor] 建议:")
            print(f"  收敛评估: {analysis['convergence_assessment']}")
            print(f"  耦合证据: {analysis['coupling_evidence']}")
            print(f"  推荐gamma: {analysis['recommended_gamma']:.4f} (当前: {current_gamma:.4f})")
            print(f"  调整比例: {analysis['adjustment_ratio']:.2f}x")
            print(f"  置信度: {analysis['confidence']}")
            print(f"  推理: {analysis['reasoning'][:100]}...")
            
            return analysis
            
        except Exception as e:
            print(f"  警告: LLM分析失败 - {e}")
            improvement_rate = (f_min_history[-1] - f_min_history[-2]) / abs(f_min_history[-2]) if len(f_min_history) >= 2 else 0
            fallback_gamma = current_gamma * (1 + 0.1 * improvement_rate)
            fallback_gamma = np.clip(fallback_gamma, 0.01, 1.0)
            
            return {
                'recommended_gamma': fallback_gamma,
                'confidence': 'medium',
                'reasoning': f'LLM不可用,使用公式: gamma * (1 + 0.1 * {improvement_rate:.4f})',
                'expected_impact': 'Standard adaptive adjustment'
            }


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试LLM Gamma智能顾问")
    print("="*70)
    
    advisor = LLMGammaAdvisor(llm_model="gpt-3.5-turbo")
    
    optimization_history = [
        {'params': {'current1': 5.5, 'charging_number': 12.0, 'current2': 2.2}, 'target': -61.0},
        {'params': {'current1': 3.5, 'charging_number': 20.0, 'current2': 1.3}, 'target': -89.0},
        {'params': {'current1': 4.8, 'charging_number': 14.0, 'current2': 2.6}, 'target': -51.0},
        {'params': {'current1': 5.0, 'charging_number': 8.0, 'current2': 3.5}, 'target': -43.0},
        {'params': {'current1': 4.2, 'charging_number': 18.0, 'current2': 1.5}, 'target': -74.0},
        {'params': {'current1': 4.4, 'charging_number': 9.5, 'current2': 2.7}, 'target': -54.0},
        {'params': {'current1': 4.8, 'charging_number': 10.9, 'current2': 4.0}, 'target': -40.0},
        {'params': {'current1': 5.7, 'charging_number': 10.0, 'current2': 4.0}, 'target': -44.0},
    ]
    
    gamma_history = [0.300, 0.302, 0.302]
    f_min_history = [-43.0, -40.0, -40.0]
    current_gamma = 0.302
    
    print("\n调用LLM分析...")
    print("-"*70)
    
    recommendation = advisor.get_gamma_recommendation(
        optimization_history=optimization_history,
        current_gamma=current_gamma,
        gamma_history=gamma_history,
        f_min_history=f_min_history
    )
    
    print("\n" + "="*70)
    print("完整建议:")
    print("="*70)
    import json
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)