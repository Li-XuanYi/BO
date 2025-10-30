"""
Warm Start生成器
整合LLM客户端、Prompt模板和响应解析器
"""

# 修改导入方式
try:
    # 作为包的一部分导入
    from .llm_client import LLMClient
    from .prompt_templates import get_warm_start_prompt, get_system_message_warm_start
    from .response_parser import ResponseParser
except ImportError:
    # 直接运行时的导入
    from llm_client import LLMClient
    from prompt_templates import get_warm_start_prompt, get_system_message_warm_start
    from response_parser import ResponseParser

from typing import List, Dict


class WarmStartGenerator:
    """
    使用LLM生成Warm Start初始点
    """
    
    def __init__(
        self,
        pbounds: Dict[str, tuple],
        model_name: str = "gpt-3.5-turbo"  # 修改默认模型
    ):
        """
        初始化生成器
        
        参数:
            pbounds: 参数边界
            model_name: LLM模型名称
        """
        self.pbounds = pbounds
        self.llm_client = LLMClient(model_name=model_name)
        self.parser = ResponseParser(pbounds=pbounds)
    
    def generate(self, n_points: int = 5) -> List[Dict[str, float]]:
        """
        生成初始点
        
        参数:
            n_points: 需要生成的点数
            
        返回:
            参数字典列表
        """
        print(f"\n开始生成{n_points}个Warm Start初始点...")
        print("-"*70)
        
        # 1. 构建Prompt
        prompt = get_warm_start_prompt(n_points=n_points)
        system_message = get_system_message_warm_start()
        
        print("正在查询LLM...")
        
        # 2. 调用LLM
        response = self.llm_client.query(
            prompt=prompt,
            system_message=system_message,
            return_json=True
        )
        
        print("LLM响应已接收，正在解析...")
        
        # 3. 解析响应
        points = self.parser.parse_warm_start_response(
            response=response,
            n_expected=n_points
        )
        
        # 4. 显示结果
        print(self.parser.format_point_summary(points))
        
        return points


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试Warm Start生成器")
    print("="*70)
    
    # 定义参数边界（与BO_demo.py一致）
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    # 创建生成器（使用gpt-3.5-turbo）
    generator = WarmStartGenerator(
        pbounds=pbounds,
        model_name="gpt-3.5-turbo"
    )
    
    # 生成初始点
    initial_points = generator.generate(n_points=5)
    
    print("\n生成完成!")
    print("="*70)
    print(f"共生成 {len(initial_points)} 个初始点")
    print("\n详细信息:")
    for i, point in enumerate(initial_points, 1):
        print(f"\n点 {i}:")
        print(f"  current1 = {point['current1']:.4f}")
        print(f"  charging_number = {point['charging_number']:.4f}")
        print(f"  current2 = {point['current2']:.4f}")