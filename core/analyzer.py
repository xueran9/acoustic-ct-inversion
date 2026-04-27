"""数据分析模块：统计、模式检测、结论生成"""
import numpy as np


class DataAnalyzer:
    """采集实验数据、计算统计指标、检测物理规律、生成实验结论"""
    def __init__(self):
        self.v_true = None
        self.sino = None
        self.vp = None
        self.loss_history = []
        self.acoustic_params = None
        self.phantom_stats = {}
        self.sino_stats = {}
        self.inversion_stats = {}
        self.patterns = []
        self.conclusion = ""
        self._stats_cache = None

    def set_phantom(self, v_true):
        self.v_true = v_true
        self.vp = None
        self.loss_history = []
        self._stats_cache = None

    def set_sino(self, sino, acoustic_params=None):
        self.sino = sino
        self.acoustic_params = acoustic_params
        self._stats_cache = None

    def set_inversion(self, vp, loss_history):
        self.vp = vp
        self.loss_history = loss_history or []
        self._stats_cache = None

    def clear(self):
        self.v_true = None
        self.sino = None
        self.vp = None
        self.loss_history = []
        self.acoustic_params = None
        self.phantom_stats = {}
        self.sino_stats = {}
        self.inversion_stats = {}
        self.patterns = []
        self.conclusion = ""
        self._stats_cache = None

    def has_phantom(self): return self.v_true is not None
    def has_sino(self): return self.sino is not None
    def has_inversion(self): return self.vp is not None

    def compute_phantom_stats(self):
        stats = {}
        if not self.has_phantom():
            return stats
        v = self.v_true
        stats['v_min'] = float(np.min(v))
        stats['v_max'] = float(np.max(v))
        stats['v_mean'] = float(np.mean(v))
        stats['v_std'] = float(np.std(v))
        gy, gx = np.gradient(v)
        stats['grad_mean'] = float(np.mean(np.sqrt(gx**2 + gy**2)))
        stats['anomaly_pct'] = float(np.mean(np.abs(v - 1500) > 50) * 100)
        stats['has_high_speed'] = bool(np.any(v > 1550))
        stats['has_low_speed'] = bool(np.any(v < 1450))
        stats['physically_valid'] = bool(np.all((v >= 1000) & (v <= 2000)))
        self.phantom_stats = stats
        return stats

    def compute_sino_stats(self):
        stats = {}
        if not self.has_sino():
            return stats
        s = self.sino
        stats['s_min'] = float(np.min(s))
        stats['s_max'] = float(np.max(s))
        stats['s_mean'] = float(np.mean(s))
        stats['s_std'] = float(np.std(s))
        stats['dynamic_range'] = float(np.max(s) - np.min(s))
        stats['snr_estimate'] = float(np.mean(s) / np.std(s)) if np.std(s) > 1e-10 else 0
        stats['is_positive'] = bool(np.all(s > 0))
        corrs = []
        for i in range(s.shape[0] - 1):
            row1 = s[i, :]
            row2 = s[i+1, :]
            if np.std(row1) > 1e-10 and np.std(row2) > 1e-10:
                c = np.corrcoef(row1, row2)[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        stats['mean_cross_corr'] = float(np.mean(corrs)) if corrs else 0
        self.sino_stats = stats
        return stats

    def compute_inversion_stats(self):
        stats = {}
        if not self.has_inversion() or not self.has_phantom():
            return stats
        vp = self.vp
        vt = self.v_true
        stats['vp_min'] = float(np.min(vp))
        stats['vp_max'] = float(np.max(vp))
        stats['vp_mean'] = float(np.mean(vp))
        stats['vp_std'] = float(np.std(vp))
        diff = vp - vt
        stats['rmse'] = float(np.sqrt(np.mean(diff**2)))
        stats['mae'] = float(np.mean(np.abs(diff)))
        stats['relative_error_pct'] = float(np.mean(np.abs(diff)) / np.mean(vt) * 100)
        if self.loss_history and len(self.loss_history) > 1:
            stats['loss_initial'] = float(self.loss_history[0])
            stats['loss_final'] = float(self.loss_history[-1])
            stats['loss_decay_ratio'] = float(self.loss_history[-1] / max(self.loss_history[0], 1e-10))
            stats['loss_converged'] = stats['loss_decay_ratio'] < 0.1
        else:
            stats['loss_initial'] = 0
            stats['loss_final'] = 0
            stats['loss_decay_ratio'] = 1.0
            stats['loss_converged'] = False
        stats['reconstruction_valid'] = bool(np.all((vp >= 1000) & (vp <= 2000)))
        self.inversion_stats = stats
        return stats

    def compute_all_stats(self):
        self.compute_phantom_stats()
        self.compute_sino_stats()
        self.compute_inversion_stats()
        self._stats_cache = {
            **self.phantom_stats,
            **self.sino_stats,
            **self.inversion_stats,
        }
        return self._stats_cache

    def detect_patterns(self, stats=None):
        if stats is None:
            stats = self._stats_cache or self.compute_all_stats()
        patterns = []

        if not self.has_phantom():
            patterns.append("⚠ 未检测到声速场数据，请先生成模拟声场。")
            self.patterns = patterns
            return patterns

        v_min = stats.get('v_min', 0)
        v_max = stats.get('v_max', 0)
        if stats.get('physically_valid', False):
            patterns.append(f"✓ 声速范围检查: 通过 ({v_min:.0f}-{v_max:.0f} m/s，在1000-2000范围内)")
        else:
            patterns.append(f"✗ 声速范围检查: 异常 ({v_min:.0f}-{v_max:.0f} m/s，超出生物组织典型范围)")

        if stats.get('has_high_speed', False) and stats.get('has_low_speed', False):
            patterns.append("✓ 非均匀性检测: 存在高速区和低速区，体模具有良好对比度")
        elif stats.get('has_high_speed', False) or stats.get('has_low_speed', False):
            patterns.append("~ 非均匀性检测: 仅存在单一类型异常区域，对比度有限")
        else:
            patterns.append("~ 非均匀性检测: 体模近似均匀，缺乏明显对比度")

        grad = stats.get('grad_mean', 0)
        if grad < 50:
            patterns.append(f"✓ 场平滑度检查: 良好 (梯度均值={grad:.2f})")
        else:
            patterns.append(f"~ 场平滑度检查: 边缘较锐利 (梯度均值={grad:.2f})")

        if self.has_sino():
            if stats.get('is_positive', False):
                patterns.append("✓ 投影正值检查: 通过 (所有走时值 > 0)")
            else:
                patterns.append("✗ 投影正值检查: 异常 (存在非正值，物理不合理的走时)")
            snr = stats.get('snr_estimate', 0)
            if snr > 10:
                patterns.append(f"✓ 投影信噪比: 优秀 (SNR={snr:.1f})")
            elif snr > 3:
                patterns.append(f"~ 投影信噪比: 一般 (SNR={snr:.1f})")
            else:
                patterns.append(f"✗ 投影信噪比: 较差 (SNR={snr:.1f})")
            corr = stats.get('mean_cross_corr', 0)
            if corr > 0.8:
                patterns.append(f"✓ 角度一致性: 良好 (相邻角度互相关={corr:.3f})")
            else:
                patterns.append(f"~ 角度一致性: 较低 (相邻角度互相关={corr:.3f})")

        if self.has_inversion():
            rmse = stats.get('rmse', 999)
            if rmse < 15:
                patterns.append(f"✓ 反演精度: 优秀 (RMSE={rmse:.2f} m/s)")
            elif rmse < 30:
                patterns.append(f"~ 反演精度: 良好 (RMSE={rmse:.2f} m/s)")
            elif rmse < 50:
                patterns.append(f"~ 反演精度: 一般 (RMSE={rmse:.2f} m/s)")
            else:
                patterns.append(f"✗ 反演精度: 较差 (RMSE={rmse:.2f} m/s)")
            if stats.get('loss_converged', False):
                patterns.append(f"✓ 损失收敛检查: 已收敛 (下降比={stats['loss_decay_ratio']:.4f})")
            else:
                patterns.append(f"~ 损失收敛检查: 未充分收敛 (下降比={stats['loss_decay_ratio']:.4f})")
            rel_err = stats.get('relative_error_pct', 999)
            if rel_err < 1:
                patterns.append(f"✓ 相对误差: 优秀 ({rel_err:.2f}%)")
            elif rel_err < 3:
                patterns.append(f"~ 相对误差: 可接受 ({rel_err:.2f}%)")
            else:
                patterns.append(f"✗ 相对误差: 偏高 ({rel_err:.2f}%)")

        if self.acoustic_params:
            freq = self.acoustic_params.get('frequency', 0)
            wtype = self.acoustic_params.get('wave_type', '未知')
            if freq > 150000:
                patterns.append(f"~ 声波参数: 频率偏高 ({freq/1000:.0f} kHz)，可能导致严重衰减")
            elif freq < 5000:
                patterns.append(f"~ 声波参数: 频率偏低 ({freq/1000:.0f} kHz)，可能分辨率不足")
            else:
                patterns.append(f"✓ 声波参数: {wtype} {freq/1000:.0f} kHz，处于合理范围")

        self.patterns = patterns
        return patterns

    def generate_conclusion(self, stats=None, patterns=None):
        if stats is None:
            stats = self._stats_cache or self.compute_all_stats()
        if patterns is None:
            patterns = self.patterns or self.detect_patterns(stats)

        parts = []
        if self.has_phantom():
            v_min = stats.get('v_min', 0)
            v_max = stats.get('v_max', 0)
            v_mean = stats.get('v_mean', 0)
            anomalies = []
            if stats.get('has_high_speed'): anomalies.append("高速区(1600 m/s)")
            if stats.get('has_low_speed'): anomalies.append("低速区(1400 m/s)")
            anom_str = "、".join(anomalies) if anomalies else "无明显异常区域"
            parts.append(f"本次实验模拟了包含{anom_str}的非均匀声速场，声速范围{v_min:.0f}-{v_max:.0f} m/s，均值{v_mean:.0f} m/s。")
        else:
            parts.append("本次实验未生成模拟声场数据。")

        if self.has_sino():
            snr = stats.get('snr_estimate', 0)
            s_range = stats.get('dynamic_range', 0)
            freq = self.acoustic_params.get('frequency', 50000) if self.acoustic_params else 50000
            wtype = self.acoustic_params.get('wave_type', '正弦波') if self.acoustic_params else '正弦波'
            parts.append(f"采用{freq/1000:.0f} kHz {wtype}作为激励信号，投影数据动态范围为{s_range:.6f} s，信噪比{snr:.1f}。")
            if snr > 10:
                parts.append("投影质量良好，信噪比充裕。")
            elif snr > 3:
                parts.append("投影质量可接受，信噪比适中。")
            else:
                parts.append("投影质量较差，建议调整声波参数以提高信噪比。")

        if self.has_inversion():
            rmse = stats.get('rmse', 0)
            mae = stats.get('mae', 0)
            loss_final = stats.get('loss_final', 0)
            epochs = len(self.loss_history) if self.loss_history else 200
            parts.append(f"AI反演经过{epochs}轮训练，最终损失值{loss_final:.6f}。")
            parts.append(f"反演结果与真实声速场的RMSE为{rmse:.2f} m/s，MAE为{mae:.2f} m/s。")
            if rmse < 15:
                parts.append("反演精度高，能够清晰分辨异常区域的位置和形状。")
            elif rmse < 30:
                parts.append("反演精度良好，异常区域轮廓可辨识。")
            elif rmse < 50:
                parts.append("反演精度一般，异常区域边界可能较为模糊。")
            else:
                parts.append("反演精度不足，建议优化网络结构、增加训练轮数或调整声波参数。")
            parts.append(f"结论：本次{'成功' if rmse < 30 else '基本'}实现了声波CT反演成像，"
                        f"反演精度{'优秀' if rmse < 15 else '良好' if rmse < 30 else '一般' if rmse < 50 else '待提高'}。")
        else:
            parts.append("结论：尚未进行AI反演，无法评估成像效果。")

        self.conclusion = "\n".join(parts)
        return self.conclusion

    def get_report_dict(self):
        stats = self._stats_cache or self.compute_all_stats()
        patterns = self.patterns or self.detect_patterns(stats)
        conclusion = self.conclusion or self.generate_conclusion(stats, patterns)
        return {
            'stats': stats,
            'patterns': patterns,
            'conclusion': conclusion,
            'acoustic_params': self.acoustic_params,
            'has_phantom': self.has_phantom(),
            'has_sino': self.has_sino(),
            'has_inversion': self.has_inversion(),
        }

    def get_csv_rows(self):
        rows = []
        report = self.get_report_dict()
        stats = report['stats']
        if self.has_phantom():
            rows.append(("声速场", "最小值", f"{stats.get('v_min',0):.2f} m/s",
                         "正常" if stats.get('physically_valid') else "异常"))
            rows.append(("声速场", "最大值", f"{stats.get('v_max',0):.2f} m/s",
                         "正常" if stats.get('physically_valid') else "异常"))
            rows.append(("声速场", "均值", f"{stats.get('v_mean',0):.2f} m/s", "-"))
            rows.append(("声速场", "标准差", f"{stats.get('v_std',0):.2f} m/s", "-"))
            rows.append(("声速场", "梯度均值", f"{stats.get('grad_mean',0):.2f}", "-"))
            anomaly_pct = stats.get('anomaly_pct', 0)
            rows.append(("声速场", "异常区域占比", f"{anomaly_pct:.1f}%",
                         "较高" if anomaly_pct > 30 else "正常"))
        if self.has_sino():
            rows.append(("投影数据", "最小值", f"{stats.get('s_min',0):.6f} s",
                         "正常" if stats.get('is_positive') else "异常"))
            rows.append(("投影数据", "最大值", f"{stats.get('s_max',0):.6f} s", "-"))
            rows.append(("投影数据", "动态范围", f"{stats.get('dynamic_range',0):.6f} s", "-"))
            snr = stats.get('snr_estimate', 0)
            snr_rating = "优秀" if snr > 10 else ("一般" if snr > 3 else "较差")
            rows.append(("投影数据", "信噪比(SNR)", f"{snr:.2f}", snr_rating))
            corr = stats.get('mean_cross_corr', 0)
            rows.append(("投影数据", "角度一致性", f"{corr:.4f}", "良好" if corr > 0.8 else "较低"))
        if self.has_inversion():
            rows.append(("反演结果", "最小值", f"{stats.get('vp_min',0):.2f} m/s", "-"))
            rows.append(("反演结果", "最大值", f"{stats.get('vp_max',0):.2f} m/s", "-"))
            rmse = stats.get('rmse', 0)
            rmse_rating = "优秀" if rmse < 15 else ("良好" if rmse < 30 else ("一般" if rmse < 50 else "较差"))
            rows.append(("反演结果", "RMSE", f"{rmse:.4f} m/s", rmse_rating))
            rows.append(("反演结果", "MAE", f"{stats.get('mae',0):.4f} m/s", "-"))
            rows.append(("反演结果", "相对误差", f"{stats.get('relative_error_pct',0):.2f}%", "-"))
            rows.append(("反演结果", "损失终值", f"{stats.get('loss_final',0):.6f}",
                         "已收敛" if stats.get('loss_converged') else "未收敛"))
            rows.append(("反演结果", "损失下降比", f"{stats.get('loss_decay_ratio',0):.4f}",
                         "良好" if stats.get('loss_decay_ratio',1) < 0.1 else "不足"))
        return rows
