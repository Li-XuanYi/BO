"""
充电工具函数模块
提供充电相关的核心计算函数
"""

import numpy as np
from SPM import SPM


def charging_time_compute(current1, charging_number, current2):
    """
    两阶段充电目标函数
    
    参数:
        current1: 第一阶段充电电流 (A)
        charging_number: 第一阶段充电步数
        current2: 第二阶段充电电流 (A)
    
    返回:
        -i: 负的总充电步数 (BO框架中越大越好)
    """
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    while not done:
        if i < int(charging_number):
            current = current1
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        else:
            current = current2
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        
        _, done, _ = env.step(current)
        i += 1
        
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 1
        
        if done:
            return -i


def record_charging_process(current1, charging_number, current2):
    """
    记录充电过程的完整时间序列数据
    
    参数:
        current1: 第一阶段充电电流 (A)
        charging_number: 第一阶段充电步数
        current2: 第二阶段充电电流 (A)
    
    返回:
        dict: 包含完整充电过程数据
            - time: 时间序列 (分钟)
            - soc: SOC历史 (%)
            - voltage: 电压历史 (V)
            - current: 电流历史 (A)
            - temperature: 温度历史 (摄氏度)
            - total_steps: 总充电步数
            - total_time: 总充电时间 (分钟)
    """
    env = SPM(3.0, 298)
    
    time_steps = []
    soc_history = []
    voltage_history = []
    current_history = []
    temp_history = []
    
    done = False
    i = 0
    t = 0
    
    while not done:
        if i < int(charging_number):
            current = current1
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        else:
            current = current2
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        
        # 记录数据
        time_steps.append(t)
        soc_history.append(env.soc * 100)
        voltage_history.append(env.voltage)
        current_history.append(current)
        temp_history.append(env.temp - 273.15)
        
        _, done, _ = env.step(current)
        i += 1
        t += env.sett['sample_time'] / 60
        
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 1
        
        if done:
            break
    
    return {
        'time': np.array(time_steps),
        'soc': np.array(soc_history),
        'voltage': np.array(voltage_history),
        'current': np.array(current_history),
        'temperature': np.array(temp_history),
        'total_steps': i,
        'total_time': t
    }


if __name__ == "__main__":
    # 自测
    print("="*70)
    print("充电工具函数模块自测")
    print("="*70)
    
    # 测试参数
    test_current1 = 5.0
    test_charging_number = 15
    test_current2 = 3.0
    
    print(f"\n测试参数:")
    print(f"  current1 = {test_current1} A")
    print(f"  charging_number = {test_charging_number}")
    print(f"  current2 = {test_current2} A")
    
    # 测试charging_time_compute
    print(f"\n测试 charging_time_compute()...")
    result = charging_time_compute(test_current1, test_charging_number, test_current2)
    print(f"  返回值: {result}")
    print(f"  充电步数: {-result}")
    
    # 测试record_charging_process
    print(f"\n测试 record_charging_process()...")
    data = record_charging_process(test_current1, test_charging_number, test_current2)
    print(f"  总步数: {data['total_steps']}")
    print(f"  总时间: {data['total_time']:.2f} 分钟")
    print(f"  最终SOC: {data['soc'][-1]:.2f}%")
    print(f"  最终电压: {data['voltage'][-1]:.3f} V")
    print(f"  最终温度: {data['temperature'][-1]:.2f} °C")
    print(f"  数据点数: {len(data['time'])}")
    
    print("\n自测完成!")
