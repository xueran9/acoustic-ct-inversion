"""虚拟实验室设备模型：波源/传感器/体模 参数映射"""
import numpy as np

# ======================================================================
# 设备定义
# ======================================================================

WAVE_SOURCES = {
    "压电换能器": {
        "freq_range": (20000, 200000),
        "freq_default": 50000,
        "wave_type": "正弦波",
        "snr_factor": 1.0,
        "stability": "高",
        "description": "压电陶瓷激励，频率20-200kHz，信号稳定，适用于常规CT成像",
        "icon": "📡",
    },
    "激光超声": {
        "freq_range": (500000, 5000000),
        "freq_default": 1000000,
        "wave_type": "脉冲波",
        "snr_factor": 0.7,
        "stability": "中",
        "description": "脉冲激光激发超声波，频率0.5-5MHz，高分辨率但信噪比较低",
        "icon": "🔦",
    },
    "电磁超声(EMAT)": {
        "freq_range": (50000, 500000),
        "freq_default": 200000,
        "wave_type": "方波",
        "snr_factor": 0.5,
        "stability": "低",
        "description": "电磁感应激发，非接触式，50-500kHz，适合高温/粗糙表面",
        "icon": "🧲",
    },
}

SENSOR_ARRAYS = {
    "线阵(128)": {
        "na": 90,
        "ns": 128,
        "scan_angle": 180,
        "geometry": "线性",
        "description": "128通道线性排列，90角度扇扫，标准工业CT配置",
        "icon": "📏",
    },
    "环阵(256)": {
        "na": 180,
        "ns": 256,
        "scan_angle": 360,
        "geometry": "环形",
        "description": "256通道环绕排列，360°全周采集，最高数据密度",
        "icon": "⭕",
    },
    "相控阵(64)": {
        "na": 64,
        "ns": 64,
        "scan_angle": 120,
        "geometry": "相控",
        "description": "64通道电子扫描，角度可调(60-180°)，支持波束聚焦",
        "icon": "📐",
    },
}

PHANTOM_TYPES = {
    "简单体模": {
        "func": "simple",
        "description": "2个异常区域（高/低声速各一），适合基础教学实验",
        "anomaly_count": 2,
        "icon": "🔵",
    },
    "复杂体模": {
        "func": "complex",
        "description": "5个随机分布的异常区域，模拟真实复杂结构",
        "anomaly_count": 5,
        "icon": "🟣",
    },
    "多层体模": {
        "func": "layered",
        "description": "4层组织结构（骨骼/肌肉/脂肪/皮肤）+肿瘤，模拟人体截面",
        "anomaly_count": 5,
        "icon": "🟢",
    },
    "肿瘤体模": {
        "func": "tumor",
        "description": "恶性肿瘤模型（高密度核心+水肿环），医学成像研究",
        "anomaly_count": 2,
        "icon": "🔴",
    },
    "自定义体模": {
        "func": "custom",
        "description": "手动配置异常区参数（位置/大小/声速）",
        "anomaly_count": 0,
        "icon": "⚙",
    },
}


# ======================================================================
# 设备效果计算
# ======================================================================

def compute_equipment_effects(wave_source, sensor_array, frequency, voltage=1.0):
    """根据设备选择计算对数据质量的影响

    Returns:
        dict with keys: snr_db, attenuation_factor, resolution_mm, na, ns, scan_angle
    """
    ws = WAVE_SOURCES.get(wave_source, WAVE_SOURCES["压电换能器"])
    sa = SENSOR_ARRAYS.get(sensor_array, SENSOR_ARRAYS["线阵(128)"])

    # SNR 估计 (dB)
    base_snr = 30  # 基准SNR
    snr_db = base_snr * ws["snr_factor"] + 10 * np.log10(voltage / 1.0 + 0.01)
    snr_db = max(5, min(60, snr_db))

    # 衰减系数
    freq_norm = frequency / ws["freq_default"]
    attenuation = 1.0 / (1 + freq_norm**1.5)

    # 分辨率估计 (mm)，频率越高分辨率越好
    wavelength = 1500 / max(frequency, 1000)  # 波长 (m)
    resolution_mm = wavelength * 1000 * 2  # Nyquist 极限 * 2

    return {
        "snr_db": round(snr_db, 1),
        "attenuation_factor": round(attenuation, 4),
        "resolution_mm": round(resolution_mm, 2),
        "na": sa["na"],
        "ns": sa["ns"],
        "scan_angle": sa["scan_angle"],
        "wave_type": ws["wave_type"],
        "frequency": int(frequency),
        "voltage": voltage,
    }


def get_equipment_summary(wave_source, sensor_array, phantom_type, frequency, voltage):
    """生成设备配置摘要文本"""
    effects = compute_equipment_effects(wave_source, sensor_array, frequency, voltage)
    ws = WAVE_SOURCES.get(wave_source, WAVE_SOURCES["压电换能器"])
    sa = SENSOR_ARRAYS.get(sensor_array, SENSOR_ARRAYS["线阵(128)"])
    pt = PHANTOM_TYPES.get(phantom_type, PHANTOM_TYPES["简单体模"])

    lines = [
        f"🔬 实验器材配置报告",
        f"{'─' * 40}",
        f"",
        f"【波源】{ws['icon']} {wave_source}",
        f"  • 类型: {ws['wave_type']}",
        f"  • 频率: {frequency/1000:.1f} kHz",
        f"  • 电压: {voltage:.1f} V",
        f"  • 稳定性: {ws['stability']}",
        f"  • {ws['description']}",
        f"",
        f"【传感器】{sa['icon']} {sensor_array}",
        f"  • 几何: {sa['geometry']}",
        f"  • 投影角度: {effects['na']}",
        f"  • 探测器数: {effects['ns']}",
        f"  • 扫描角度: {sa['scan_angle']}°",
        f"  • {sa['description']}",
        f"",
        f"【体模】{pt['icon']} {phantom_type}",
        f"  • 异常区数: {pt['anomaly_count']}",
        f"  • {pt['description']}",
        f"",
        f"【预估数据质量】",
        f"  • 信噪比: {effects['snr_db']:.1f} dB",
        f"  • 分辨率: {effects['resolution_mm']:.2f} mm",
        f"  • 衰减系数: {effects['attenuation_factor']:.4f}",
        f"  • 数据量: {effects['na']}×{effects['ns']} = {effects['na']*effects['ns']} 采样点",
    ]
    return "\n".join(lines)


def build_acoustic_params(wave_source, frequency, voltage):
    """根据设备选择构建 acoustic_params dict"""
    ws = WAVE_SOURCES.get(wave_source, WAVE_SOURCES["压电换能器"])
    return {
        'frequency': int(frequency),
        'amplitude': voltage,
        'wave_type': ws['wave_type'],
        'external': None,
    }
