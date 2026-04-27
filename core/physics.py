"""物理模拟模块：声场生成与正演投影"""
import numpy as np
from scipy.ndimage import gaussian_filter


def gen_phantom(size=64):
    """生成模拟非均匀声速场（体模）"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    v = 1500 * np.ones((size, size), dtype=np.float64)
    r1 = np.sqrt((x + 0.3) ** 2 + (y - 0.2) ** 2) < 0.25
    r2 = np.sqrt((x - 0.4) ** 2 + (y + 0.3) ** 2) < 0.2
    v[r1] = 1600
    v[r2] = 1400
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
