"""物理模拟模块：声场生成与正演投影"""
import numpy as np
from scipy.ndimage import gaussian_filter


def gen_phantom(size=64):
    """默认体模（简单体模）"""
    return gen_phantom_simple(size)


def gen_phantom_simple(size=64):
    """简单体模：2个异常区（高/低各一）"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    v = 1500 * np.ones((size, size), dtype=np.float64)
    r1 = np.sqrt((x + 0.3) ** 2 + (y - 0.2) ** 2) < 0.25
    r2 = np.sqrt((x - 0.4) ** 2 + (y + 0.3) ** 2) < 0.2
    v[r1] = 1600
    v[r2] = 1400
    return gaussian_filter(v, 1.5)


def gen_phantom_complex(size=64, seed=None):
    """复杂体模：5个随机大小/位置/声速的异常区"""
    if seed is not None:
        np.random.seed(seed)
    v = 1500 * np.ones((size, size), dtype=np.float64)
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    anomalies = []
    attempts = 0
    while len(anomalies) < 5 and attempts < 200:
        attempts += 1
        cx = np.random.uniform(-0.7, 0.7)
        cy = np.random.uniform(-0.7, 0.7)
        radius = np.random.uniform(0.08, 0.2)
        speed = np.random.choice([1350, 1400, 1420, 1580, 1600, 1650])
        too_close = any(np.sqrt((cx - a[0])**2 + (cy - a[1])**2) < (radius + a[2] + 0.05)
                       for a in anomalies)
        if not too_close:
            anomalies.append((cx, cy, radius, speed))
    for cx, cy, r, sp in anomalies:
        mask = np.sqrt((x - cx)**2 + (y - cy)**2) < r
        v[mask] = sp
    return gaussian_filter(v, 1.2)


def gen_phantom_layered(size=64):
    """多层体模：模拟组织分层结构（皮肤/脂肪/肌肉/骨骼）"""
    v = 1500 * np.ones((size, size), dtype=np.float64)
    y_vals = np.linspace(-1, 1, size)
    # 四层结构
    v[y_vals < -0.4] = 1650    # 底层：骨骼
    v[(y_vals >= -0.4) & (y_vals < -0.1)] = 1580  # 肌肉层
    v[(y_vals >= -0.1) & (y_vals < 0.3)] = 1450   # 脂肪层
    v[y_vals >= 0.3] = 1520    # 皮肤层
    # 添加一个圆形肿瘤
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    tumor = np.sqrt((x - 0.2)**2 + (y - 0.05)**2) < 0.15
    v[tumor] = 1620
    return gaussian_filter(v, 0.8)


def gen_phantom_tumor(size=64):
    """肿瘤体模：高密度核心+低密度环（模拟恶性肿瘤）"""
    v = 1500 * np.ones((size, size), dtype=np.float64)
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # 高密度肿瘤核心
    core = np.sqrt((x - 0.15)**2 + (y + 0.1)**2) < 0.18
    v[core] = 1620
    # 低密度环（水肿带）
    ring = (np.sqrt((x - 0.15)**2 + (y + 0.1)**2) < 0.25) & ~core
    v[ring] = 1380
    # 第二个小肿瘤
    core2 = np.sqrt((x + 0.5)**2 + (y - 0.4)**2) < 0.1
    v[core2] = 1600
    return gaussian_filter(v, 1.0)


def gen_phantom_custom(size=64, anomalies=None):
    """自定义体模：用户指定异常区列表

    anomalies: [(cx, cy, radius, speed), ...] 每个异常区的中心坐标(-1~1)、半径、声速
    """
    v = 1500 * np.ones((size, size), dtype=np.float64)
    if not anomalies:
        return v
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    for cx, cy, radius, speed in anomalies:
        mask = np.sqrt((x - cx)**2 + (y - cy)**2) < radius
        v[mask] = speed
    return gaussian_filter(v, 1.5)


def forward_proj(vel, na=90, ns=128, acoustic_params=None):
    """正演投影：计算声波走时投影数据（正弦图）"""
    size = vel.shape[0]
    sino = np.zeros((na, ns), dtype=np.float64)

    attenuation_factor = 1.0
    if acoustic_params is not None:
        freq = acoustic_params.get('frequency', 50000)
        attenuation_factor = 1.0 / (1 + (freq / 100000) ** 2)

    for i, th in enumerate(np.linspace(0, np.pi, na, endpoint=False)):
        c, s = np.cos(th), np.sin(th)
        for j in range(ns):
            t = np.linspace(-1, 1, 200)
            xr = c * t - s * (2 * j / ns - 1)
            yr = s * t + c * (2 * j / ns - 1)
            xi = ((xr + 1) / 2) * (size - 1)
            yi = ((yr + 1) / 2) * (size - 1)

            x0 = np.floor(xi).astype(int)
            x1 = x0 + 1
            y0 = np.floor(yi).astype(int)
            y1 = y0 + 1

            x0 = np.clip(x0, 0, size - 1)
            x1 = np.clip(x1, 0, size - 1)
            y0 = np.clip(y0, 0, size - 1)
            y1 = np.clip(y1, 0, size - 1)

            wa = (x1 - xi) * (y1 - yi)
            wb = (xi - x0) * (y1 - yi)
            wc = (x1 - xi) * (yi - y0)
            wd = (xi - x0) * (yi - y0)

            v_line = wa * vel[y0, x0] + wb * vel[y0, x1] + wc * vel[y1, x0] + wd * vel[y1, x1]
            v_line = np.clip(v_line, 1000, 2000)
            ds = 2 / 200
            sino[i, j] = np.sum(1 / v_line) * ds * attenuation_factor

    sino = np.nan_to_num(sino, nan=0.001, posinf=0.002, neginf=0.001)
    return sino
