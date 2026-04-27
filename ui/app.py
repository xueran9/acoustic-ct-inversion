"""主窗口：ActCTApp — 整合流水线、声波采集、分析、AI助手"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile
import matplotlib
matplotlib.use('TkAgg')

from core.analyzer import DataAnalyzer
from core.assistant import AIAssistant
from core.collector import WaveCollector
from core.physics import gen_phantom, forward_proj
from core.model import Net
from core.wave_generator import generate_acoustic_wave
from utils.exporters import export_txt_report, export_csv_report
from .theme import apply_light_theme, LIGHT_THEME
from .experiment_page import ExperimentPage
from .wave_page import WavePage
from .analysis_page import AnalysisPage
from .assistant_page import AssistantPage
from .widgets import PENDING, RUNNING, COMPLETED, FAILED


class ActCTApp(tk.Tk):
    """主窗口 — 驱动整个应用的核心控制器"""
    def __init__(self):
        super().__init__()
        self.title("AI驱动声学CT反演实验系统")
        self.geometry("1200x800")
        self.minsize(900, 600)

        # 应用浅色主题
        self.theme = apply_light_theme(self)

        # 实验核心数据
        self.size = 64
        self.net = None
        self.v_true = None
        self.sino = None
        self.vp = None
        self.loss_history = []
        self.cbars = []

        # 声波参数变量
        self.wave_freq = tk.DoubleVar(value=50000)
        self.wave_amp = tk.DoubleVar(value=1.0)
        self.wave_type = tk.StringVar(value="正弦波")
        self.external_wave = None

        # 核心服务
        self.analyzer = DataAnalyzer()
        self.assistant = AIAssistant(self.analyzer)
        self.wave_collector = WaveCollector()
        self.experiment_history = []
        self.exp_counter = 0

        # 主笔记本
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 构建页面
        self.page_experiment = ExperimentPage(self.notebook, self)
        self.notebook.add(self.page_experiment, text="CT反演实验")

        self.page_wave = WavePage(self.notebook, self)
        self.notebook.add(self.page_wave, text="声波输入口")

        self.page_analysis = AnalysisPage(self.notebook, self)
        self.notebook.add(self.page_analysis, text="实验数据分析")

        self.page_assistant = AssistantPage(self.notebook, self)
        self.notebook.add(self.page_assistant, text="AI实验助手")

        self.reset_pipeline()

    # ===================== 流水线控制 =====================

    def set_status(self, s):
        self.page_experiment.lbl_status.config(text=f"状态：{s}")
        self.update()

    def pipeline_step_complete(self, idx):
        self.page_experiment.pipeline.set_step_state(idx, COMPLETED)

    def pipeline_step_failed(self, idx):
        self.page_experiment.pipeline.set_step_state(idx, FAILED)

    def pipeline_step_running(self, idx):
        self.page_experiment.pipeline.set_step_state(idx, RUNNING)

    def run_single_step(self, idx):
        """执行指定流水线步骤"""
        try:
            if idx == 0:
                self.pipeline_step_running(0)
                self.set_status("正在生成模拟声场...")
                self.make_phantom()
                self.pipeline_step_complete(0)
                self.page_experiment.pipeline.set_status("声场已生成")
            elif idx == 1:
                if self.v_true is None:
                    messagebox.showwarning("提示", "请先生成模拟声场！")
                    return
                self.pipeline_step_running(1)
                self.set_status("正在计算声波投影...")
                self.make_sino()
                self.pipeline_step_complete(1)
                self.page_experiment.pipeline.set_status("投影已完成")
            elif idx == 2:
                if self.sino is None:
                    messagebox.showwarning("提示", "请先生成声波投影！")
                    return
                self.pipeline_step_running(2)
                self.set_status("正在AI反演...")
                self.run_inv()
                self.pipeline_step_complete(2)
                self.page_experiment.pipeline.set_status("反演已完成")
            elif idx == 3:
                self.pipeline_step_running(3)
                self.set_status("正在分析数据...")
                self.refresh_analysis()
                self.assistant.refresh()
                self.pipeline_step_complete(3)
                self.page_experiment.pipeline.set_status("分析已完成")
            elif idx == 4:
                self.pipeline_step_running(4)
                self.save_experiment_snapshot()
                self.pipeline_step_complete(4)
                self.page_experiment.pipeline.set_status("全部完成！")
                self.set_status("实验流水线已完成")
        except Exception as e:
            self.pipeline_step_failed(idx)
            self.page_experiment.pipeline.set_status(f"步骤{idx+1}失败")
            messagebox.showerror("错误", f"步骤{idx+1}执行失败：{str(e)}")

    def run_all_steps(self):
        """一键运行全部流水线步骤"""
        self.reset_pipeline()
        self.after(100, self._run_all_sequence)

    def _run_all_sequence(self):
        for i in range(5):
            self.run_single_step(i)
            if self.page_experiment.pipeline.get_step_state(i) == FAILED:
                break
            self.update()

    def reset_pipeline(self):
        """重置流水线状态和实验数据"""
        self.net = None
        self.v_true = None
        self.sino = None
        self.vp = None
        self.loss_history = []
        self.external_wave = None
        self.analyzer.clear()
        self.assistant.clear_chat()
        self.page_experiment.pipeline.reset_all()
        if hasattr(self, 'page_experiment'):
            self.page_experiment._update_plots()
        if hasattr(self, 'page_analysis'):
            self.refresh_analysis()
        self.set_status("已重置，等待操作")

    # ===================== 核心实验 =====================

    def make_phantom(self):
        self.v_true = gen_phantom(self.size)
        self.page_experiment._update_plots()
        self.analyzer.set_phantom(self.v_true)
        self.refresh_analysis()
        self.set_status("模拟声场已生成")

    def make_sino(self):
        if self.v_true is None:
            messagebox.showwarning("提示", "请先生成模拟声场！")
            return
        acoustic_params = {
            'frequency': self.wave_freq.get(),
            'amplitude': self.wave_amp.get(),
            'wave_type': self.wave_type.get(),
            'external': self.external_wave
        }
        self.sino = forward_proj(self.v_true, acoustic_params=acoustic_params)
        self.page_experiment._update_plots()
        self.analyzer.set_sino(self.sino, acoustic_params)
        self.refresh_analysis()
        self.set_status(f"投影生成完成，范围：[{np.min(self.sino):.6f}, {np.max(self.sino):.6f}] 秒")

    def run_inv(self):
        if self.sino is None:
            messagebox.showwarning("提示", "请先生成声波投影！")
            return
        self.set_status("AI反演中，请稍候...")
        self.net = Net(self.size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        st = torch.FloatTensor(self.sino).unsqueeze(0)
        vt = torch.FloatTensor(self.v_true).unsqueeze(0)
        epochs = 200
        self.loss_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.net(st)
            loss = criterion(output, vt)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if epoch % 20 == 0:
                self.set_status(f"AI反演中... 第 {epoch}/{epochs} 轮，损失：{loss.item():.6f}")
        self.vp = self.net(st).detach().numpy().squeeze()
        self.page_experiment._update_plots()
        self.analyzer.set_inversion(self.vp, self.loss_history)
        self.auto_analyze_on_completion()
        self.set_status("反演完成！")

    def auto_analyze_on_completion(self):
        self.analyzer.compute_all_stats()
        self.analyzer.detect_patterns()
        self.analyzer.generate_conclusion()
        self.refresh_analysis()
        self.assistant.refresh()
        self.save_experiment_snapshot()

    # ===================== 声波采集 =====================

    def start_collection(self):
        target = 5
        self.wave_collector.start_collection(target)
        self.page_wave.refresh_ui()
        self.page_wave.update_log()

    def stop_collection(self):
        if not self.wave_collector.is_collecting:
            return
        count = len(self.wave_collector.accepted_waves)
        if count == 0 and not messagebox.askyesno("确认", "尚无有效声波，确定要结束吗？"):
            return
        self.wave_collector.stop_early()
        self.page_wave.refresh_ui()
        self.page_wave.update_log()
        if count > 0:
            self._on_collection_complete()
        else:
            messagebox.showwarning("提示", "采集结束，但无有效声波数据。")

    def reset_collection(self):
        if messagebox.askyesno("确认", "确定要清空所有采集数据吗？"):
            self.wave_collector.clear()
            self.page_wave.refresh_ui()
            self.page_wave.update_log()

    def submit_wave_for_collection(self):
        if not self.wave_collector.is_collecting:
            messagebox.showinfo("提示", "请先点击「开始采集」启动采集会话。")
            return
        _, wave_data = generate_acoustic_wave(
            frequency=self.wave_freq.get(),
            amplitude=self.wave_amp.get(),
            wave_type=self.wave_type.get()
        )
        acoustic_params = {
            'frequency': self.wave_freq.get(),
            'amplitude': self.wave_amp.get(),
            'wave_type': self.wave_type.get(),
            'external': self.external_wave
        }
        result = self.wave_collector.submit_wave(wave_data, acoustic_params, override=False)
        if not result['accepted'] and result.get('type_mismatch'):
            if messagebox.askyesno("类型不匹配", f"{result['reason']}\n\n是否强制接受此声波？"):
                result = self.wave_collector.submit_wave(wave_data, acoustic_params, override=True)
        self.page_wave.refresh_ui()
        self.page_wave.update_log()
        if result.get('complete'):
            self._on_collection_complete()

    def _on_collection_complete(self):
        if not self.wave_collector.accepted_waves:
            return
        unified = self.wave_collector.unify_waves()
        if unified:
            self.wave_freq.set(unified.get('frequency', 50000))
            self.wave_amp.set(unified.get('amplitude', 1.0))
            if unified.get('wave_type'):
                self.wave_type.set(unified['wave_type'])
        self.set_status("采集完成，自动执行实验...")
        # 从采集完成自动执行流水线
        self.make_phantom()
        self.make_sino()
        self.run_inv()
        self.set_status(f"采集实验完成 (共 {len(self.wave_collector.accepted_waves)} 次声波)")
        self.notebook.select(self.page_analysis)

    def preview_wave(self):
        t, wave = generate_acoustic_wave(
            frequency=self.wave_freq.get(),
            amplitude=self.wave_amp.get(),
            wave_type=self.wave_type.get()
        )
        title = f"声波波形预览 ({self.wave_type.get()}, {self.wave_freq.get() / 1000:.1f} kHz)"
        self.page_wave.update_wave_preview(t, wave, title)

    def load_wav_file(self):
        file_path = filedialog.askopenfilename(
            title="选择声波文件",
            filetypes=[("WAV音频文件", "*.wav"), ("所有文件", "*.*")]
        )
        if not file_path:
            return
        try:
            sample_rate, data = wavfile.read(file_path)
            if len(data.shape) > 1:
                data = data[:, 0]
            data = data / np.max(np.abs(data))
            self.external_wave = data
            messagebox.showinfo("成功", f"成功加载声波文件！\n采样率: {sample_rate} Hz\n长度: {len(data)} 点")
            self.page_wave.update_wave_preview_from_data(data, "外部加载的声波波形")
        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败：{str(e)}")

    def apply_wave_params(self):
        self.preview_wave()
        messagebox.showinfo("成功", "声波参数已应用！\n请返回【CT反演实验】页面生成投影。")
        self.notebook.select(self.page_experiment)

    # ===================== 实验数据分析 =====================

    def refresh_analysis(self):
        """刷新实验数据分析页面的所有显示"""
        analyzer = self.analyzer
        page = self.page_analysis

        page.clear_stats_table()
        if not analyzer.has_phantom() and not analyzer.has_sino() and not analyzer.has_inversion():
            page.add_stat_row("提示", "", "尚无实验数据", "请先执行实验")
            page.set_patterns_text("尚无实验数据，请先完成CT反演实验。")
            page.set_conclusion_text("尚无实验数据，请先完成CT反演实验。")
            return

        stats = analyzer.compute_all_stats()
        patterns = analyzer.detect_patterns(stats)
        conclusion = analyzer.generate_conclusion(stats, patterns)

        if analyzer.has_phantom():
            page.add_stat_row("声速场", "最小值", f"{stats.get('v_min',0):.2f} m/s",
                              "正常" if stats.get('physically_valid') else "异常")
            page.add_stat_row("声速场", "最大值", f"{stats.get('v_max',0):.2f} m/s",
                              "正常" if stats.get('physically_valid') else "异常")
            page.add_stat_row("声速场", "均值", f"{stats.get('v_mean',0):.2f} m/s", "-")
            page.add_stat_row("声速场", "标准差", f"{stats.get('v_std',0):.2f} m/s", "-")
            page.add_stat_row("声速场", "梯度均值", f"{stats.get('grad_mean',0):.2f}", "-")
            anomaly_pct = stats.get('anomaly_pct', 0)
            page.add_stat_row("声速场", "异常区域占比", f"{anomaly_pct:.1f}%",
                              "较高" if anomaly_pct > 30 else "正常")

        if analyzer.has_sino():
            page.add_stat_row("投影数据", "最小值", f"{stats.get('s_min',0):.6f} s",
                              "正常" if stats.get('is_positive') else "异常")
            page.add_stat_row("投影数据", "最大值", f"{stats.get('s_max',0):.6f} s", "-")
            page.add_stat_row("投影数据", "动态范围", f"{stats.get('dynamic_range',0):.6f} s", "-")
            snr = stats.get('snr_estimate', 0)
            snr_rating = "优秀" if snr > 10 else ("一般" if snr > 3 else "较差")
            page.add_stat_row("投影数据", "信噪比(SNR)", f"{snr:.2f}", snr_rating)
            corr = stats.get('mean_cross_corr', 0)
            page.add_stat_row("投影数据", "角度一致性", f"{corr:.4f}", "良好" if corr > 0.8 else "较低")

        if analyzer.has_inversion():
            page.add_stat_row("反演结果", "最小值", f"{stats.get('vp_min',0):.2f} m/s", "-")
            page.add_stat_row("反演结果", "最大值", f"{stats.get('vp_max',0):.2f} m/s", "-")
            rmse = stats.get('rmse', 0)
            rmse_rating = "优秀" if rmse < 15 else ("良好" if rmse < 30 else ("一般" if rmse < 50 else "较差"))
            page.add_stat_row("反演结果", "RMSE", f"{rmse:.4f} m/s", rmse_rating)
            page.add_stat_row("反演结果", "MAE", f"{stats.get('mae',0):.4f} m/s", "-")
            page.add_stat_row("反演结果", "相对误差", f"{stats.get('relative_error_pct',0):.2f}%", "-")
            page.add_stat_row("反演结果", "损失终值", f"{stats.get('loss_final',0):.6f}",
                              "已收敛" if stats.get('loss_converged') else "未收敛")
            page.add_stat_row("反演结果", "损失下降比", f"{stats.get('loss_decay_ratio',0):.4f}",
                              "良好" if stats.get('loss_decay_ratio',1) < 0.1 else "不足")

        pattern_text = "\n".join(patterns) if patterns else "未检测到模式信息。"
        page.set_patterns_text(pattern_text)
        page.set_conclusion_text(conclusion)
        page.refresh_visualization(self.loss_history, self.vp, self.v_true)

    # ===================== 报告导出 =====================

    def export_txt_report(self):
        if not (self.analyzer.has_phantom() or self.analyzer.has_sino() or self.analyzer.has_inversion()):
            messagebox.showwarning("提示", "尚无实验数据，请先执行CT反演实验。")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            title="导出实验报告"
        )
        if not path:
            return
        try:
            export_txt_report(self.analyzer, self.assistant, path)
            messagebox.showinfo("成功", f"实验报告已导出至：\n{path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{str(e)}")

    def export_csv_report(self):
        if not (self.analyzer.has_phantom() or self.analyzer.has_sino() or self.analyzer.has_inversion()):
            messagebox.showwarning("提示", "尚无实验数据，请先执行CT反演实验。")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            title="导出统计数据"
        )
        if not path:
            return
        try:
            export_csv_report(self.analyzer, path)
            messagebox.showinfo("成功", f"统计数据已导出至：\n{path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{str(e)}")

    # ===================== 实验历史记录 =====================

    def save_experiment_snapshot(self):
        if not (self.analyzer.has_phantom() or self.analyzer.has_sino() or self.analyzer.has_inversion()):
            return
        self.exp_counter += 1
        snapshot = {
            'id': self.exp_counter,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'acoustic_params': dict(self.analyzer.acoustic_params) if self.analyzer.acoustic_params else {},
            'phantom_stats': dict(self.analyzer.phantom_stats),
            'sino_stats': dict(self.analyzer.sino_stats),
            'inversion_stats': dict(self.analyzer.inversion_stats),
            'patterns': list(self.analyzer.patterns),
            'conclusion': self.analyzer.conclusion,
            'score': self.assistant.experiment_score or 0,
            'rating': self.assistant.quality_rating or "无数据",
            'v_true': np.copy(self.v_true) if self.v_true is not None else None,
            'sino': np.copy(self.sino) if self.sino is not None else None,
            'vp': np.copy(self.vp) if self.vp is not None else None,
            'loss_history': list(self.loss_history),
        }
        self.experiment_history.insert(0, snapshot)
        if len(self.experiment_history) > 50:
            self.experiment_history = self.experiment_history[:50]
        self.page_analysis.refresh_history_list(self.experiment_history)

    def load_selected_experiment(self):
        selection = self.page_analysis.history_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请先在列表中选择一个实验。")
            return
        snap_id = int(selection[0])
        snap = next((s for s in self.experiment_history if s['id'] == snap_id), None)
        if snap is None:
            return
        if not messagebox.askyesno("确认", "加载历史实验将覆盖当前数据，是否继续？"):
            return
        self.net = None
        self.v_true = np.copy(snap['v_true']) if snap['v_true'] is not None else None
        self.sino = np.copy(snap['sino']) if snap['sino'] is not None else None
        self.vp = np.copy(snap['vp']) if snap['vp'] is not None else None
        self.loss_history = list(snap['loss_history'])
        self.analyzer.clear()
        if self.v_true is not None:
            self.analyzer.set_phantom(self.v_true)
        if self.sino is not None:
            self.analyzer.set_sino(self.sino, dict(snap['acoustic_params']))
        if self.vp is not None:
            self.analyzer.set_inversion(self.vp, self.loss_history)
        self.analyzer.phantom_stats = dict(snap['phantom_stats'])
        self.analyzer.sino_stats = dict(snap['sino_stats'])
        self.analyzer.inversion_stats = dict(snap['inversion_stats'])
        self.analyzer.patterns = list(snap['patterns'])
        self.analyzer.conclusion = snap['conclusion']
        self.page_experiment._update_plots()
        self.refresh_analysis()
        self.assistant.refresh()
        self.set_status(f"已加载历史实验 #{snap['id']} ({snap['timestamp']})")

    def open_compare_window(self):
        if len(self.experiment_history) < 2:
            messagebox.showwarning("提示", "至少需要保存2个实验才能进行比较。")
            return
        win = tk.Toplevel(self)
        win.title("实验比较")
        win.geometry("1000x600")
        win.configure(bg='#f5f6fa')

        T = self.theme
        sel_frame = ttk.Frame(win, padding=5)
        sel_frame.pack(fill=tk.X)
        ttk.Label(sel_frame, text="实验A:").pack(side=tk.LEFT)
        combo_a = ttk.Combobox(sel_frame, state="readonly", width=30)
        combo_a.pack(side=tk.LEFT, padx=5)
        ttk.Label(sel_frame, text="实验B:").pack(side=tk.LEFT, padx=(20, 0))
        combo_b = ttk.Combobox(sel_frame, state="readonly", width=30)
        combo_b.pack(side=tk.LEFT, padx=5)

        labels = [f"#{s['id']} {s['timestamp']} (评分:{s['score']})" for s in self.experiment_history]
        combo_a['values'] = labels
        combo_b['values'] = labels
        if len(labels) >= 2:
            combo_a.current(0)
            combo_b.current(1)

        display_frame = ttk.Frame(win, padding=5)
        display_frame.pack(fill=tk.BOTH, expand=True)

        text_a = tk.Text(display_frame, wrap=tk.WORD, font=("微软雅黑", 9),
                         bg='#ffffff', fg='#1a1a2e')
        text_b = tk.Text(display_frame, wrap=tk.WORD, font=("微软雅黑", 9),
                         bg='#ffffff', fg='#1a1a2e')
        text_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        text_b.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        scroll_a = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=text_a.yview)
        scroll_b = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=text_b.yview)
        text_a.configure(yscrollcommand=scroll_a.set)
        text_b.configure(yscrollcommand=scroll_b.set)
        scroll_a.pack(side=tk.LEFT, fill=tk.Y)
        scroll_b.pack(side=tk.RIGHT, fill=tk.Y)

        def update_comparison(event=None):
            idx_a = combo_a.current()
            idx_b = combo_b.current()
            if idx_a < 0 or idx_b < 0:
                return
            snap_a = self.experiment_history[idx_a]
            snap_b = self.experiment_history[idx_b]
            for text_w, snap in [(text_a, snap_a), (text_b, snap_b)]:
                text_w.delete(1.0, tk.END)
                p = snap['acoustic_params']
                inv = snap['inversion_stats']
                lines = [
                    f"实验 #{snap['id']} - {snap['timestamp']}",
                    f"评分: {snap['score']}/100 ({snap['rating']})",
                    "=" * 35,
                    f"波形: {p.get('wave_type','?')}  频率: {p.get('frequency',0)/1000:.0f} kHz",
                    f"振幅: {p.get('amplitude',1.0):.2f}",
                    "",
                    "【反演精度】",
                    f"  RMSE: {inv.get('rmse',0):.2f} m/s",
                    f"  MAE: {inv.get('mae',0):.2f} m/s",
                    f"  相对误差: {inv.get('relative_error_pct',0):.2f}%",
                    f"  损失终值: {inv.get('loss_final',0):.6f}",
                    f"  收敛: {'是' if inv.get('loss_converged') else '否'}",
                    "",
                    "【检测结论】",
                ]
                for pat in snap['patterns']:
                    lines.append(f"  {pat}")
                text_w.insert(tk.END, "\n".join(lines))
        combo_a.bind("<<ComboboxSelected>>", update_comparison)
        combo_b.bind("<<ComboboxSelected>>", update_comparison)
        update_comparison()

        btn_frame = ttk.Frame(win, padding=5)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="关闭", command=win.destroy).pack(side=tk.RIGHT)
        ttk.Label(btn_frame, text="选择两个实验进行对比分析").pack(side=tk.LEFT)

    def clear_history(self):
        if not self.experiment_history:
            return
        if messagebox.askyesno("确认", "确定要清空所有历史记录吗？此操作不可恢复。"):
            self.experiment_history.clear()
            self.exp_counter = 0
            self.page_analysis.refresh_history_list(self.experiment_history)
            self.page_analysis.update_history_detail("历史记录已清空。")
