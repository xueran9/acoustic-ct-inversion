"""声波采集管理器：验证合理性、类型一致性检查、统一处理"""
import time
import numpy as np
from collections import Counter


class WaveCollector:
    """声波采集管理器：验证合理性、类型一致性检查、统一处理"""
    FREQ_MIN = 5000
    FREQ_MAX = 200000
    FREQ_WARN_HIGH = 150000
    FREQ_WARN_LOW = 5000
    AMP_MIN = 0.1
    AMP_MAX = 10.0
    FREQ_TOLERANCE = 0.10
    WAVE_TYPES = ["正弦波", "方波", "脉冲波"]
    SNR_MIN = 3.0

    def __init__(self):
        self.target_count = 5
        self.accepted_waves = []
        self.rejected_waves = []
        self.wave_type = None
        self.is_collecting = False
        self.collection_complete = False
        self._current_index = 0
        self._session_log = []
        self._last_submit_time = 0

    def start_collection(self, target_count=5):
        self.clear()
        self.target_count = max(1, int(target_count))
        self.is_collecting = True
        self.collection_complete = False
        self._add_log("info", f"采集会话开始，目标接收 {self.target_count} 次")

    def validate_wave_data(self, wave_data, acoustic_params):
        issues = []
        warnings = []
        if wave_data is None:
            issues.append("声波数据为空")
            return {'valid': False, 'issues': issues, 'warnings': warnings}
        if np.any(np.isnan(wave_data)) or np.any(np.isinf(wave_data)):
            issues.append("声波数据包含无效值(NaN/Inf)")
        freq = acoustic_params.get('frequency', 0)
        if freq < self.FREQ_MIN or freq > self.FREQ_MAX:
            issues.append(f"频率 {freq/1000:.0f}kHz 超出合理范围 ({self.FREQ_MIN/1000:.0f}-{self.FREQ_MAX/1000:.0f} kHz)")
        elif freq > self.FREQ_WARN_HIGH:
            warnings.append(f"频率偏高 ({freq/1000:.0f} kHz)，可能导致严重衰减")
        elif freq < self.FREQ_WARN_LOW:
            warnings.append(f"频率偏低 ({freq/1000:.0f} kHz)，可能分辨率不足")
        amp = acoustic_params.get('amplitude', 0)
        if amp < self.AMP_MIN or amp > self.AMP_MAX:
            warnings.append(f"振幅 {amp:.2f} 超出推荐范围 ({self.AMP_MIN}-{self.AMP_MAX})")
        wtype = acoustic_params.get('wave_type', '')
        if wtype not in self.WAVE_TYPES:
            issues.append(f"未知波形类型: {wtype}")
        if acoustic_params.get('external') is not None:
            wav_data = acoustic_params['external']
            if np.max(np.abs(wav_data)) >= 0.99:
                warnings.append("WAV信号可能存在削波(接近最大振幅)")
            if len(wav_data) > 100:
                noise_floor = np.std(wav_data[-len(wav_data)//10:])
                signal_rms = np.std(wav_data)
                snr = signal_rms / max(noise_floor, 1e-10)
                if snr < self.SNR_MIN:
                    warnings.append(f"WAV信噪比较低 ({snr:.1f})")
        return {'valid': len(issues) == 0, 'issues': issues, 'warnings': warnings}

    def check_same_type(self, new_params, reference_params):
        if reference_params is None:
            return {'same': True, 'reason': '', 'can_override': False}
        new_type = new_params.get('wave_type', '')
        ref_type = reference_params.get('wave_type', '')
        new_freq = new_params.get('frequency', 0)
        ref_freq = reference_params.get('frequency', 0)
        reasons = []
        if new_type != ref_type:
            reasons.append(f"波形类型不同 ({ref_type} vs {new_type})")
        if ref_freq > 0 and abs(new_freq - ref_freq) / ref_freq > self.FREQ_TOLERANCE:
            reasons.append(f"频率差异过大 ({ref_freq/1000:.0f}kHz vs {new_freq/1000:.0f}kHz)")
        is_same = len(reasons) == 0
        return {'same': is_same, 'reason': "；".join(reasons), 'can_override': not is_same}

    def submit_wave(self, wave_data, acoustic_params, override=False):
        now = time.time()
        if now - self._last_submit_time < 0.5:
            return {'accepted': False, 'reason': '操作过于频繁，请稍后再试', 'count': len(self.accepted_waves), 'complete': False}
        self._last_submit_time = now
        if not self.is_collecting or self.collection_complete:
            return {'accepted': False, 'reason': '采集未启动或已完成', 'count': len(self.accepted_waves), 'complete': False}
        result = {'accepted': False, 'reason': '', 'count': len(self.accepted_waves), 'complete': False}
        self._current_index += 1
        v = self.validate_wave_data(wave_data, acoustic_params)
        if not v['valid']:
            entry = {'index': self._current_index, 'wave_data': wave_data, 'acoustic_params': dict(acoustic_params),
                     'reason': '；'.join(v['issues']), 'timestamp': time.strftime('%H:%M:%S')}
            self.rejected_waves.append(entry)
            self._add_log("reject", f"第{self._current_index}次: {'；'.join(v['issues'])}")
            result['reason'] = '；'.join(v['issues'])
            return result
        if self.accepted_waves:
            ref = self.accepted_waves[0]['acoustic_params']
            tc = self.check_same_type(acoustic_params, ref)
            if not tc['same'] and not override:
                entry = {'index': self._current_index, 'wave_data': wave_data, 'acoustic_params': dict(acoustic_params),
                         'reason': tc['reason'], 'timestamp': time.strftime('%H:%M:%S')}
                self.rejected_waves.append(entry)
                self._add_log("warning", f"第{self._current_index}次: 类型不匹配 - {tc['reason']}")
                result['reason'] = tc['reason']
                result['type_mismatch'] = True
                return result
        entry = {'index': self._current_index, 'wave_data': wave_data, 'acoustic_params': dict(acoustic_params),
                 'timestamp': time.strftime('%H:%M:%S')}
        self.accepted_waves.append(entry)
        if len(self.accepted_waves) == 1:
            self.wave_type = {'wave_type': acoustic_params.get('wave_type', ''), 'frequency': acoustic_params.get('frequency', 0)}
        log_msg = f"第{self._current_index}次: {acoustic_params.get('wave_type','')} {acoustic_params.get('frequency',0)/1000:.0f}kHz"
        if v['warnings']:
            log_msg += " (" + "；".join(v['warnings']) + ")"
        self._add_log("accept", log_msg)
        complete = len(self.accepted_waves) >= self.target_count
        if complete:
            self.collection_complete = True
            self.is_collecting = False
            self._add_log("complete", f"采集完成！共接收 {len(self.accepted_waves)} 次")
        return {'accepted': True, 'reason': log_msg, 'count': len(self.accepted_waves),
                'complete': complete, 'target': self.target_count, 'warnings': v['warnings']}

    def stop_early(self):
        if not self.is_collecting:
            return
        self.is_collecting = False
        self.collection_complete = True
        count = len(self.accepted_waves)
        if count > 0:
            self._add_log("complete", f"采集提前结束，共接收 {count} 次")
        else:
            self._add_log("warning", "提前结束（无有效声波）")

    def unify_waves(self):
        if not self.accepted_waves:
            return {}
        base = dict(self.accepted_waves[0]['acoustic_params'])
        if len(self.accepted_waves) > 1:
            freqs = [w['acoustic_params'].get('frequency', 50000) for w in self.accepted_waves]
            amps = [w['acoustic_params'].get('amplitude', 1.0) for w in self.accepted_waves]
            types = [w['acoustic_params'].get('wave_type', '') for w in self.accepted_waves]
            most_common_type = Counter(types).most_common(1)[0][0]
            base['frequency'] = float(np.mean(freqs))
            base['amplitude'] = float(np.mean(amps))
            base['wave_type'] = most_common_type
        base['collection_count'] = len(self.accepted_waves)
        base['unified'] = True
        return base

    def get_results(self):
        if not self.accepted_waves and not self.rejected_waves:
            return "尚无采集记录。"
        lines = [f"采集目标: {self.target_count} 次", f"已接受: {len(self.accepted_waves)} 次", f"已拒绝: {len(self.rejected_waves)} 次"]
        if self.accepted_waves:
            lines.append("")
            lines.append("【接受记录】")
            for w in self.accepted_waves:
                p = w['acoustic_params']
                lines.append(f"  #{w['index']} {p.get('wave_type','?')} {p.get('frequency',0)/1000:.0f}kHz")
        if self.rejected_waves:
            lines.append("")
            lines.append("【拒绝记录】")
            for w in self.rejected_waves:
                lines.append(f"  #{w['index']} {w['reason']}")
        return "\n".join(lines)

    def get_log_text(self):
        if not self._session_log:
            return "尚无采集记录。\n请点击「开始采集」启动声波采集会话。"
        lines = []
        icons = {'accept': '✓', 'reject': '✗', 'warning': '⚠', 'info': '●', 'complete': '★'}
        for entry in self._session_log:
            icon = icons.get(entry['type'], '●')
            lines.append(f"[{entry['timestamp']}] {icon} {entry['message']}")
        return "\n".join(lines)

    def _add_log(self, log_type, message):
        self._session_log.append({'timestamp': time.strftime('%H:%M:%S'), 'type': log_type, 'message': message})

    def clear(self):
        self.accepted_waves = []
        self.rejected_waves = []
        self.wave_type = None
        self.is_collecting = False
        self.collection_complete = False
        self._current_index = 0
        self._session_log = []
        self._last_submit_time = 0
