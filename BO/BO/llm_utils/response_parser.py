"""
LLM响应解析器
解析并验证LLM返回的参数集
"""

import json
from typing import List, Dict, Any, Optional


class ResponseParser:
    """
    解析LLM返回的JSON响应
    """
    
    def __init__(self, pbounds: Dict[str, tuple]):
        """
        初始化解析器
        
        参数:
            pbounds: 参数边界字典，例如 {'current1': (3, 6), ...}
        """
        self.pbounds = pbounds
        self.required_fields = ['current1', 'charging_number', 'current2']
    
    def parse_warm_start_response(
        self, 
        response: str, 
        n_expected: int = 5
    ) -> List[Dict[str, float]]:
        """
        解析Warm Start的LLM响应
        
        参数:
            response: LLM返回的文本
            n_expected: 期望得到的点数量
            
        返回:
            参数字典列表，例如 [{'current1': 4.5, 'charging_number': 12, 'current2': 2.0}, ...]
        """
        try:
            # 尝试直接解析JSON
            data = self._extract_json(response)
            
            # 验证数据格式
            if not isinstance(data, list):
                raise ValueError(f"期望返回list，但得到 {type(data)}")
            
            # 提取并验证每个参数集
            valid_points = []
            for i, item in enumerate(data):
                try:
                    point = self._validate_point(item, point_index=i)
                    valid_points.append(point)
                except Exception as e:
                    print(f"警告: 第{i+1}个点验证失败 - {e}")
                    continue
            
            # 检查是否得到足够的点
            if len(valid_points) < n_expected:
                print(f"警告: 只得到{len(valid_points)}个有效点，期望{n_expected}个")
            
            return valid_points[:n_expected]
            
        except json.JSONDecodeError as e:
            print(f"错误: JSON解析失败 - {e}")
            print("原始响应:")
            print(response)
            raise
        except Exception as e:
            print(f"错误: 响应解析失败 - {e}")
            raise
    
    def _extract_json(self, text: str) -> Any:
        """
        从文本中提取JSON
        支持：
        - 纯JSON
        - Markdown代码块中的JSON
        """
        text = text.strip()
        
        # 情况1: 纯JSON
        if text.startswith('[') or text.startswith('{'):
            return json.loads(text)
        
        # 情况2: Markdown代码块
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            json_text = text[start:end].strip()
            return json.loads(json_text)
        
        # 情况3: 普通代码块
        if '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            json_text = text[start:end].strip()
            return json.loads(json_text)
        
        # 尝试直接解析
        return json.loads(text)
    
    def _validate_point(
        self, 
        point: Dict[str, Any], 
        point_index: int
    ) -> Dict[str, float]:
        """
        验证单个参数点
        
        参数:
            point: 参数字典
            point_index: 点的索引（用于错误提示）
            
        返回:
            验证后的参数字典
        """
        validated = {}
        
        # 检查必需字段
        for field in self.required_fields:
            if field not in point:
                raise ValueError(f"缺少必需字段: {field}")
            
            value = point[field]
            
            # 转换为float
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"字段 {field} 的值 {value} 无法转换为数字")
            
            # 检查边界
            if field in self.pbounds:
                lower, upper = self.pbounds[field]
                if value < lower or value > upper:
                    print(f"警告: {field}={value} 超出边界 [{lower}, {upper}]，将截断")
                    value = max(lower, min(upper, value))
            
            validated[field] = value
        
        return validated
    
    def format_point_summary(self, points: List[Dict[str, float]]) -> str:
        """
        格式化参数点摘要（用于打印）
        
        参数:
            points: 参数字典列表
            
        返回:
            格式化的字符串
        """
        summary = "\n解析得到的初始点:\n"
        summary += "-" * 70 + "\n"
        summary += f"{'#':<4} {'current1':<12} {'charging_number':<18} {'current2':<12}\n"
        summary += "-" * 70 + "\n"
        
        for i, point in enumerate(points, 1):
            summary += f"{i:<4} {point['current1']:<12.2f} {point['charging_number']:<18.2f} {point['current2']:<12.2f}\n"
        
        summary += "-" * 70
        return summary


# 测试代码
if __name__ == "__main__":
    print("测试ResponseParser")
    print("="*70)
    
    # 定义参数边界
    pbounds = {
        'current1': (3, 6),
        'charging_number': (5, 25),
        'current2': (1, 4)
    }
    
    parser = ResponseParser(pbounds)
    
    # 测试1: 正常的JSON响应
    print("\n测试1: 正常JSON响应")
    print("-"*70)
    test_response_1 = """
[
  {
    "current1": 4.5,
    "charging_number": 12,
    "current2": 2.0,
    "strategy_name": "balanced",
    "reasoning": "平衡策略"
  },
  {
    "current1": 5.5,
    "charging_number": 15,
    "current2": 2.8,
    "strategy_name": "aggressive",
    "reasoning": "激进策略"
  }
]
"""
    points_1 = parser.parse_warm_start_response(test_response_1, n_expected=2)
    print(parser.format_point_summary(points_1))
    
    # 测试2: Markdown代码块中的JSON
    print("\n\n测试2: Markdown代码块")
    print("-"*70)
    test_response_2 = """
这是我的建议:
```json
[
  {
    "current1": 3.8,
    "charging_number": 15,
    "current2": 2.2
  },
  {
    "current1": 4.2,
    "charging_number": 10,
    "current2": 2.5
  }
]
```

希望有帮助!
"""
    points_2 = parser.parse_warm_start_response(test_response_2, n_expected=2)
    print(parser.format_point_summary(points_2))
    
    # 测试3: 超出边界的值
    print("\n\n测试3: 边界检查")
    print("-"*70)
    test_response_3 = """
[
  {
    "current1": 7.0,
    "charging_number": 12,
    "current2": 0.5
  }
]
"""
    points_3 = parser.parse_warm_start_response(test_response_3, n_expected=1)
    print(parser.format_point_summary(points_3))
    
    print("\n\n测试完成")
    print("="*70)