"""声波信号生成模块"""
import numpy as np


def generate_acoustic_wave(frequency=50000, amplitude=1.0, wave_type="正弦波", duration=0.001, sample_rate=1e6):
    """
    生成模拟声波信号
    frequency: 频率 (Hz, 默认50kHz)
    amplitude: 振幅
    wave_type: 波形类型 ("正弦波", "方波", "脉冲波")
    duration: 持续时间 (秒)
    sample_rate: 采样率
    """
    t = np.linspace(0, duration, int(sample_rate * duration))

    if wave_type == "正弦波":
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
    elif wave_type == "方波":
        wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif wave_type == "脉冲波":
        wave = np.zeros_like(t)
        pulse_center = len(t) // 2
        pulse_width = int(sample_rate * 0.0001)
        wave[pulse_center - pulse_width // 2: pulse_center + pulse_width // 2] = amplitude
    else:
        wave = amplitude * np.sin(2 * np.pi * frequency * t)

    return t, wave
