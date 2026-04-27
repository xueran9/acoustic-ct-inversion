"""页面1：实验控制（流水线模式 + 可视化）"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from .widgets import PipelinePanel, PENDING, RUNNING, COMPLETED, FAILED


class ExperimentPage(ttk.Frame):
    """实验控制页面 — 流水线面板 + 3图可视化"""
    PIPELINE_STEPS = ["生成声场", "声波投影", "AI反演", "数据分析", "完成"]

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.cbars = []
        self._build_ui()

    def _build_ui(self):
        # 流水线面板（卡片式）
        pipeline_card = ttk.LabelFrame(self, text="实验流水线", padding=8)
        pipeline_card.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.pipeline = PipelinePanel(pipeline_card)
        self.pipeline.pack(fill=tk.X)
        self.pipeline.set_steps(self.PIPELINE_STEPS)
        self.pipeline.set_on_run_step(self._on_run_step)
        self.pipeline.set_on_run_all(self._on_run_all)
        self.pipeline.set_on_reset(self._on_reset)

        # 状态栏
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, padx=12, pady=(0, 2))
        self.lbl_status = ttk.Label(status_frame, text="状态：等待操作", foreground="#2563eb")
        self.lbl_status.pack(side=tk.RIGHT)

        # 可视化区域（卡片）
        plot_card = ttk.LabelFrame(self, text="实验可视化", padding=5)
        plot_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.fig_ct = plt.figure(figsize=(12, 4.5), dpi=80)
        self.fig_ct.patch.set_facecolor('#ffffff')
        self.ax1 = self.fig_ct.add_subplot(131)
        self.ax1.set_title("真实声速场")
        self.ax2 = self.fig_ct.add_subplot(132)
        self.ax2.set_title("声波投影数据")
        self.ax3 = self.fig_ct.add_subplot(133)
        self.ax3.set_title("AI反演结果")

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.tick_params(colors='#6b7280')
            ax.grid(True, alpha=0.3)

        self.canvas_ct = FigureCanvasTkAgg(self.fig_ct, plot_card)
        self.canvas_ct.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_ct.draw()

    def _on_run_step(self):
        """执行当前步骤"""
        idx = self.pipeline.current_step_index()
        if idx < 0:
            self.app.set_status("所有步骤已完成")
            return
        self.app.run_single_step(idx)

    def _on_run_all(self):
        """一键运行全部"""
        self.app.run_all_steps()

    def _on_reset(self):
        """重置流水线"""
        self.app.reset_pipeline()

    def _update_plots(self):
        """刷新三个子图"""
        app = self.app
        for cb in self.cbars:
            try:
                cb.remove()
            except Exception:
                pass
        self.cbars = []
        self.fig_ct.patch.set_facecolor('#ffffff')
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.tick_params(colors='#6b7280')
            ax.grid(True, alpha=0.3)

        if app.v_true is not None:
            self.ax1.set_title("真实非均匀声速场")
            im = self.ax1.imshow(app.v_true, cmap="jet", origin="lower")
            self.cbars.append(plt.colorbar(im, ax=self.ax1, fraction=0.046))
        else:
            self.ax1.set_title("真实声速场")

        if app.sino is not None:
            self.ax2.set_title("声波走时投影（Sinogram）")
            im = self.ax2.imshow(app.sino, cmap="gray", aspect="auto")
            self.cbars.append(plt.colorbar(im, ax=self.ax2, fraction=0.046))
        else:
            self.ax2.set_title("声波投影数据")

        if app.vp is not None:
            self.ax3.set_title("AI反演声速场")
            im = self.ax3.imshow(app.vp, cmap="jet", origin="lower")
            self.cbars.append(plt.colorbar(im, ax=self.ax3, fraction=0.046))
        else:
            self.ax3.set_title("AI反演结果")

        self.canvas_ct.draw()
