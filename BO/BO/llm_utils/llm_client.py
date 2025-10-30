"""
LLM客户端封装
适配nuwaapi的OpenAI接口
"""

from openai import OpenAI
from typing import Dict, List, Optional
import json


class LLMClient:
    """
    LLM API调用的统一接口
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo",
        base_url: str = "https://api.nuwaapi.com/v1",
        api_key: str = "sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        初始化LLM客户端
        
        参数:
            model_name: 模型名称，默认 "gpt-4.5-turbo"
            base_url: API基础URL
            api_key: API密钥
            temperature: 生成温度（0-1，越低越确定性）
            max_tokens: 最大生成token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        print(f"LLM客户端初始化成功: {model_name}")
    
    def query(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        return_json: bool = False
    ) -> str:
        """
        向LLM发送查询
        
        参数:
            prompt: 用户提示词
            system_message: 系统提示词（可选）
            return_json: 是否要求返回JSON格式
            
        返回:
            LLM的响应文本
        """
        messages = []
        
        # 添加系统消息
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            # 默认系统消息
            messages.append({
                "role": "system", 
                "content": "You are an expert in lithium-ion battery technology and optimization."
            })
        
        # 添加用户消息
        if return_json:
            prompt += "\n\nIMPORTANT: Return your response as valid JSON."
        messages.append({"role": "user", "content": prompt})
        
        try:
            # 调用API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = completion.choices[0].message.content
            
            # 如果要求JSON格式，尝试解析
            if return_json:
                try:
                    json.loads(content)  # 验证是否为有效JSON
                except json.JSONDecodeError:
                    print("警告: LLM返回的不是有效JSON，返回原始文本")
            
            return content
            
        except Exception as e:
            print(f"错误: LLM查询失败 - {e}")
            raise
    
    def batch_query(
        self, 
        prompts: List[str], 
        system_message: Optional[str] = None
    ) -> List[str]:
        """
        批量查询（串行执行）
        
        参数:
            prompts: 提示词列表
            system_message: 系统提示词
            
        返回:
            响应列表
        """
        responses = []
        for i, prompt in enumerate(prompts):
            print(f"批量查询进度: {i+1}/{len(prompts)}")
            response = self.query(prompt, system_message)
            responses.append(response)
        return responses


# 测试代码
if __name__ == "__main__":
    print("开始测试LLM客户端...")
    
    # 创建客户端
    client = LLMClient()
    
    # 测试1: 简单查询
    print("\n" + "="*50)
    print("测试1: 简单查询")
    print("="*50)
    test_prompt = "简单介绍一下锂电池的充电原理，不超过100字。"
    response = client.query(test_prompt)
    print(f"\n提示: {test_prompt}")
    print(f"\n响应:\n{response}")
    
    # 测试2: 与充电优化相关的查询
    print("\n" + "="*50)
    print("测试2: 充电策略相关查询")
    print("="*50)
    test_prompt2 = """
对于5Ah的锂电池，从0%充电到80%，设计一个两阶段充电策略：
- 第一阶段用电流current1充电charging_number步
- 第二阶段用电流current2充电直到完成

请给出合理的参数范围建议：
- current1应该在多少安培？
- charging_number应该持续多少步？
- current2应该在多少安培？

简短回答即可。
"""
    response2 = client.query(test_prompt2)
    print(f"\n提示: {test_prompt2}")
    print(f"\n响应:\n{response2}")
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)