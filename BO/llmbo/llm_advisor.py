from llm_utils import LLMClient
import json

class LLMKernelAdvisor:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm_client = LLMClient(model_name=model_name)
    
    def analyze_gamma_adjustment(self, current_gamma, improvement_rate, 
                                 historical_results, gamma_history, f_min_history):
        prompt = f"""
        分析贝叶斯优化，建议gamma调整。
        当前gamma: {current_gamma:.4f}
        改进率: {improvement_rate:.6f}
        
        返回JSON: {{"recommended_gamma": 0.35, "confidence": "high", "reasoning": "..."}}
        """
        response = self.llm_client.query(prompt=prompt, return_json=True)
        return json.loads(response)