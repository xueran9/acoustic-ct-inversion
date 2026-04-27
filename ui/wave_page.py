"""页面2：声波输入口 — 采集管理 + 参数设置 + 波形预览"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.io import wavfile
from core.wave_generator import generate_acoustic_wave


class WavePage(ttk.Frame):
    """声波输入口页面 — 采集、参数、预览"""
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        # ========== 采集管理区 ==========
        frame_mgr = ttk.LabelFrame(self, text="声波采集管理", padding=12)
        frame_mgr.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.collection_status = ttk.Label(frame_mgr, text="状态: 未开始",
                                           font=("微软雅黑", 10, "bold"))
        self.collection_status.pack(anchor=tk.W, padx=5, pady=2)

        self.collection_info = ttk.Label(frame_mgr, text="目标: 5 | 已接受: 0 | 已拒绝: 0")
        self.collection_info.pack(anchor=tk.W, padx=5, pady=2)

        self.collection_progress = ttk.Progressbar(frame_mgr, orient=tk.HORIZONTAL,
                                                    length=400, mode='determinate')
        self.collection_progress.pack(anchor=tk.W, padx=5, pady=2)
        self.collection_prog_label = ttk.Label(frame_mgr, text="0/5")
        self.collection_prog_label.pack(anchor=tk.W, padx=5)

        btn_mgr = ttk.Frame(frame_mgr)
        btn_mgr.pack(pady=5, anchor=tk.W)
        ttk.Button(btn_mgr, text="开始采集", command=self.app.start_collection).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_mgr, text="提前结束", command=self.app.stop_collection).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_mgr, text="清空重置", command=self.app.reset_collection).pack(side=tk.LEFT, padx=3)

        # ========== 声波参数设置 ==========
        frame_params = ttk.LabelFrame(self, text="声波参数设置", padding=12)
        frame_params.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame_params, text="频率 (Hz):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame_params, textvariable=self.app.wave_freq, width=15).grid(row=0, column=1, padx=5)
        ttk.Scale(frame_params, from_=1000, to=200000, variable=self.app.wave_freq,
                  orient=tk.HORIZONTAL, length=200).grid(row=0, column=2, padx=10)

        ttk.Label(frame_params, text="振幅:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame_params, textvariable=self.app.wave_amp, width=15).grid(row=1, column=1, padx=5)

        ttk.Label(frame_params, text="波形类型:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(frame_params, textvariable=self.app.wave_type,
                     values=["正弦波", "方波", "脉冲波"],
                     state="readonly").grid(row=2, column=1, padx=5)

        btn_params = ttk.Frame(frame_params)
        btn_params.grid(row=3, column=0, columnspan=3, pady=10)
        ttk.Button(btn_params, text="预览声波波形", command=self.app.preview_wave).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_params, text="加载外部WAV文件", command=self.app.load_wav_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_params, text="应用声波参数", command=self.app.apply_wave_params).pack(side=tk.LEFT, padx=5)

        # ========== 提交声波按钮 ==========
        self.btn_submit_wave = ttk.Button(self, text="═══ 提交声波 ═══", style="Accent.TButton",
                                          command=self.app.submit_wave_for_collection)
        self.btn_submit_wave.pack(fill=tk.X, padx=10, pady=5)

        # ========== 采集日志 ==========
        frame_log = ttk.LabelFrame(self, text="采集日志", padding=8)
        frame_log.pack(fill=tk.X, padx=10, pady=5)

        self.collection_log = tk.Text(frame_log, wrap=tk.WORD, height=6,
                                      bg='#ffffff', fg='#1a1a2e',
                                      insertbackground='#1a1a2e',
                                      font=("微软雅黑", 9), state=tk.DISABLED,
                                      highlightthickness=1, highlightbackground='#e5e7eb',
                                      relief=tk.FLAT)
        scrollbar = ttk.Scrollbar(frame_log, orient=tk.VERTICAL, command=self.collection_log.yview)
        self.collection_log.configure(yscrollcommand=scrollbar.set)
        self.collection_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._update_log_display()

        # ========== 波形预览 ==========
        frame_preview = ttk.LabelFrame(self, text="波形预览", padding=5)
        frame_preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.fig_wave = plt.figure(figsize=(10, 3), dpi=80)
        self.fig_wave.patch.set_facecolor('#ffffff')
        self.ax_wave = self.fig_wave.add_subplot(111)
        self.ax_wave.set_title("声波波形预览")
        self.ax_wave.tick_params(colors='#6b7280')
        self.ax_wave.grid(True, alpha=0.3)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, frame_preview)
        self.canvas_wave.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_wave.draw()

    def refresh_ui(self):
        """刷新采集管理区的所有控件"""
        wc = self.app.wave_collector
        accepted = len(wc.accepted_waves)
        rejected = len(wc.rejected_waves)
        target = wc.target_count
        if wc.collection_complete:
            status = "已完成" if accepted >= target else "已结束"
        elif wc.is_collecting:
            status = f"收集中 ({accepted}/{target})"
        else:
            status = "未开始"
        self.collection_status.config(text=f"状态: {status}")
        self.collection_info.config(text=f"目标: {target} | 已接受: {accepted} | 已拒绝: {rejected}")
        pct = int(accepted / max(target, 1) * 100)
        self.collection_progress['value'] = min(pct, 100)
        self.collection_prog_label.config(text=f"{accepted}/{target}")

    def _update_log_display(self):
        text = self.app.wave_collector.get_log_text()
        self.collection_log.configure(state=tk.NORMAL)
        self.collection_log.delete(1.0, tk.END)
        self.collection_log.insert(tk.END, text)
        self.collection_log.configure(state=tk.DISABLED)
        self.collection_log.see(tk.END)

    def update_log(self):
        """Public method for app to call to refresh log"""
        self._update_log_display()

    def update_wave_preview(self, t, wave, title):
        """Update the wave preview plot"""
        self.ax_wave.clear()
        self.ax_wave.tick_params(colors='#6b7280')
        self.ax_wave.plot(t * 1000, wave)
        self.ax_wave.set_title(title)
        self.ax_wave.set_xlabel("时间 (ms)")
        self.ax_wave.set_ylabel("振幅")
        self.ax_wave.grid(True, alpha=0.3)
        self.canvas_wave.draw()

    def update_wave_preview_from_data(self, data, title):
        """Update preview from loaded WAV data"""
        self.ax_wave.clear()
        self.ax_wave.tick_params(colors='#6b7280')
        self.ax_wave.plot(data[:min(1000, len(data))])
        self.ax_wave.set_title(title)
        self.ax_wave.grid(True, alpha=0.3)
        self.canvas_wave.draw()
