"""
LLMBO的Prompt模板
参考论文Figure 2设计，针对锂电池充电优化问题
"""


def get_warm_start_prompt(n_points: int = 5) -> str:
    """
    生成Warm Start的Prompt
    
    参数:
        n_points: 需要生成的初始点数量
        
    返回:
        完整的prompt字符串
    """
    
    prompt = f"""
As an expert in lithium-ion battery fast-charging technology, generate {n_points} initial candidate parameter sets for a two-stage constant-current charging optimization problem.

BATTERY SPECIFICATIONS:
- Model: INR21700-M50T (Chen2020 parameter set)
- Nominal capacity: 5.0 Ah
- Voltage range: 2.5V (lower cutoff) to 4.2V (upper cutoff)
- Initial state: Voltage = 3.0V, Temperature = 298K (25°C), SOC = 0%
- Target state: SOC = 80%
- Time step: 90 seconds per step

CHARGING STRATEGY:
Two-stage constant-current protocol:
- Stage 1: Apply current1 (A) for charging_number steps
- Stage 2: Apply current2 (A) until SOC reaches 80%

OPTIMIZATION PARAMETERS AND BOUNDS:
1. current1: First-stage charging current, range [3.0, 6.0] Amperes
   - Physical meaning: Higher current → faster initial charging BUT higher heat generation
   - Typical fast-charging: 0.6C to 1.2C (3A to 6A for 5Ah battery)

2. charging_number: Duration of first stage, range [5, 25] steps
   - Physical meaning: When to switch from aggressive to conservative charging
   - Time range: 450 seconds (7.5 min) to 2250 seconds (37.5 min)

3. current2: Second-stage charging current, range [1.0, 4.0] Amperes
   - Physical meaning: Lower current for safe completion and longevity
   - Typical CV-phase equivalent: 0.2C to 0.8C (1A to 4A for 5Ah battery)

PHYSICAL CONSTRAINTS AND DOMAIN KNOWLEDGE:

1. Voltage constraint (CRITICAL):
   - Terminal voltage must not exceed 4.2V
   - When voltage ≥ 4.0V, current automatically decays: I_actual = I_set × exp(-0.9 × (V - 4.0))
   - This means aggressive early charging will trigger decay sooner

2. Temperature constraint (CRITICAL):
   - Cell temperature must not exceed 309K (36°C)
   - Higher current → more heat generation (I²R losses)
   - Prolonged high current → cumulative heating

3. Parameter coupling effects:

   a) current1 ↔ charging_number (STRONG coupling, importance: 0.7):
      - High current1 + long charging_number → Risk: overheating and early voltage limit
      - High current1 → Voltage rises faster → Should use shorter charging_number
      - Example: If current1 = 5.5A, suggest charging_number ≤ 12

   b) charging_number ↔ current2 (MODERATE coupling, importance: 0.6):
      - Short charging_number → Stage 2 does more work → May need higher current2
      - Long charging_number → Stage 2 does less work → Can use lower current2
      - Example: If charging_number = 8, consider current2 ≥ 2.0A

   c) current1 ↔ current2 (WEAK coupling, importance: 0.3):
      - Large gap (e.g., current1=6A, current2=1A) → Abrupt transition, may cause instability
      - Small gap (e.g., current1=3.5A, current2=2.5A) → Smooth but possibly suboptimal speed
      - Thermal carry-over: Heat from Stage 1 affects Stage 2 initial temperature

4. Optimization trade-offs:
   - Objective: Minimize total charging time (steps)
   - Constraint violation penalty: +1 step for each voltage or temperature violation
   - Strategy balance: Fast charging vs. constraint satisfaction vs. smooth operation

EXPERT STRATEGIES TO CONSIDER:

1. Aggressive strategy:
   - High current1 (5.0-6.0A), short charging_number (5-10), moderate current2 (2.0-2.5A)
   - Risk: May violate temperature constraint, voltage decay triggers early
   - Suitable when: Prioritizing speed over safety margins

2. Conservative strategy:
   - Moderate current1 (3.0-4.0A), long charging_number (18-25), low current2 (1.0-1.5A)
   - Benefit: Safe operation, minimal constraint violations
   - Suitable when: Prioritizing battery health and longevity

3. Balanced strategy:
   - Medium current1 (4.0-5.0A), medium charging_number (11-15), medium current2 (2.0-3.0A)
   - Benefit: Balance between speed and safety
   - Suitable when: Typical fast-charging scenario

4. Early-transition strategy:
   - Medium-high current1 (4.5-5.5A), short charging_number (6-10), higher current2 (3.2-4.0A)
   - Rationale: Quick initial boost, then distribute work to Stage 2

5. Late-transition strategy:
   - Medium current1 (4.0-4.8A), long charging_number (16-23), lower current2 (1.2-1.8A)
   - Rationale: Maximize work in Stage 1 while staying within limits

OUTPUT FORMAT:
Generate {n_points} diverse parameter sets in JSON format. Each set should:
- Represent a distinct charging strategy
- Be physically reasonable and respect constraints
- Balance exploration (diverse strategies) and exploitation (known good practices)
- Include your reasoning for each parameter choice

Return a JSON array with the following structure:
[
  {{
    "current1": <float between 3.0 and 6.0>,
    "charging_number": <float between 5 and 25>,
    "current2": <float between 1.0 and 4.0>,
    "strategy_name": "<brief name>",
    "reasoning": "<explain why these parameters make sense together>"
  }},
  ...
]

IMPORTANT: 
- Ensure all values are within the specified bounds
- Consider parameter coupling when choosing values
- Provide diverse strategies (don't generate 5 similar points)
- Think about how these parameters will affect charging time and constraint satisfaction
"""
    
    return prompt


def get_system_message_warm_start() -> str:
    """
    Warm Start的系统消息
    """
    return """You are a world-class expert in lithium-ion battery technology with deep knowledge in:
- Electrochemistry and battery physics
- Fast-charging protocols and optimization
- Battery management systems (BMS)
- Thermal management and safety constraints

Your task is to apply your domain expertise to generate intelligent initial guesses for battery charging optimization, considering both performance and safety."""


# 测试函数
if __name__ == "__main__":
    print("="*70)
    print("LLM Warm Start Prompt Template")
    print("="*70)
    
    prompt = get_warm_start_prompt(n_points=5)
    system_msg = get_system_message_warm_start()
    
    print("\nSYSTEM MESSAGE:")
    print("-"*70)
    print(system_msg)
    
    print("\n\nUSER PROMPT:")
    print("-"*70)
    print(prompt)
    
    print("\n\n")
    print("="*70)
    print("Prompt长度:", len(prompt), "字符")
    print("="*70)