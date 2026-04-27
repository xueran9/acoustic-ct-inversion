"""页面3：实验数据分析 — 统计/一致性/结论/可视化/历史"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class AnalysisPage(ttk.Frame):
    """实验数据分析页面 — 5个标签子页"""
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        # 顶部工具栏
        frame_top = ttk.LabelFrame(self, text="分析控制", padding=8)
        frame_top.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(frame_top, text="实验数据分析", font=("微软雅黑", 12, "bold")).pack(side=tk.LEFT)
        ttk.Button(frame_top, text="🔄 刷新分析", command=self.app.refresh_analysis).pack(side=tk.RIGHT, padx=3)
        ttk.Button(frame_top, text="📄 导出TXT", command=self.app.export_txt_report).pack(side=tk.RIGHT, padx=3)
        ttk.Button(frame_top, text="📊 导出CSV", command=self.app.export_csv_report).pack(side=tk.RIGHT, padx=3)

        # 子标签页
        self.analysis_notebook = ttk.Notebook(self)
        self.analysis_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 子页1：数据统计
        self.page_stats = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.page_stats, text="数据统计")
        self._build_stats_tab()

        # 子页2：一致性分析
        self.page_patterns = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.page_patterns, text="一致性分析")
        self._build_patterns_tab()

        # 子页3：实验结论
        self.page_conclusion = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.page_conclusion, text="实验结论")
        self._build_conclusion_tab()

        # 子页4：可视化
        self.page_visual = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.page_visual, text="可视化")
        self._build_visualization_tab()

        # 子页5：历史记录
        self.page_history = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.page_history, text="历史记录")
        self._build_history_tab()

    # ---------- 子页1：数据统计 ----------
    def _build_stats_tab(self):
        frame = ttk.Frame(self.page_stats, padding=5)
        frame.pack(fill=tk.BOTH, expand=True)
        columns = ("类别", "指标", "数值", "评价")
        self.stats_tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.stats_tree.heading(col, text=col)
        self.stats_tree.column("类别", width=120)
        self.stats_tree.column("指标", width=100)
        self.stats_tree.column("数值", width=140)
        self.stats_tree.column("评价", width=100)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=scrollbar.set)
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_tree.insert("", tk.END, values=("提示", "", "尚无实验数据", "请先执行CT反演实验"))

    # ---------- 子页2：一致性分析 ----------
    def _build_patterns_tab(self):
        # 卡片容器
        card = tk.Frame(self.page_patterns, bg='#f8fafc', padx=12, pady=12)
        card.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左侧强调条
        accent = tk.Frame(card, bg='#2563eb', width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 右侧内容区
        content = tk.Frame(card, bg='#f8fafc')
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header = tk.Label(content, text="一致性分析", font=("微软雅黑", 11, "bold"),
                         bg='#f8fafc', fg='#1a1a2e')
        header.pack(anchor=tk.W, pady=(0, 6))

        text_frame = tk.Frame(content, bg='#e5e7eb')
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.patterns_text = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED,
                                      font=("微软雅黑", 10), height=18,
                                      bg='#ffffff', fg='#1a1a2e',
                                      insertbackground='#1a1a2e',
                                      padx=12, pady=10,
                                      borderwidth=0, relief=tk.FLAT,
                                      highlightthickness=0)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.patterns_text.yview)
        self.patterns_text.configure(yscrollcommand=scrollbar.set)
        self.patterns_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 配置彩色标签
        self.patterns_text.tag_configure("pass", foreground="#059669")
        self.patterns_text.tag_configure("warn", foreground="#d97706")
        self.patterns_text.tag_configure("fail", foreground="#dc2626")
        self.patterns_text.tag_configure("info", foreground="#6b7280")

        self.patterns_text.configure(state=tk.NORMAL)
        self.patterns_text.insert(tk.END, "尚无实验数据\n\n", "info")
        self.patterns_text.insert(tk.END, "完成实验后点击「刷新分析」查看一致性分析。")
        self.patterns_text.configure(state=tk.DISABLED)

    # ---------- 子页3：实验结论 ----------
    def _build_conclusion_tab(self):
        # 外层卡片容器
        card = tk.Frame(self.page_conclusion, bg='#f8fafc', padx=12, pady=12)
        card.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左侧彩色强调条（蓝色）
        accent = tk.Frame(card, bg='#2563eb', width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 右侧内容区
        content = tk.Frame(card, bg='#f8fafc')
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header = tk.Label(content, text="实验结论", font=("微软雅黑", 11, "bold"),
                         bg='#f8fafc', fg='#1a1a2e')
        header.pack(anchor=tk.W, pady=(0, 6))

        # 带边框的文字容器
        text_frame = tk.Frame(content, bg='#e5e7eb')
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.conclusion_text = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED,
                                        font=("微软雅黑", 10), height=18,
                                        bg='#ffffff', fg='#1a1a2e',
                                        insertbackground='#1a1a2e',
                                        padx=12, pady=10,
                                        borderwidth=0, relief=tk.FLAT,
                                        highlightthickness=0)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.conclusion_text.yview)
        self.conclusion_text.configure(yscrollcommand=scrollbar.set)
        self.conclusion_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 配置标签样式
        self.conclusion_text.tag_configure("h1", font=("微软雅黑", 11, "bold"),
                                           foreground="#1a1a2e", spacing1=6, spacing3=2)
        self.conclusion_text.tag_configure("metric", font=("微软雅黑", 10, "bold"),
                                           foreground="#2563eb")
        self.conclusion_text.tag_configure("good", foreground="#059669",
                                           font=("微软雅黑", 10, "bold"))
        self.conclusion_text.tag_configure("ok", foreground="#d97706")
        self.conclusion_text.tag_configure("bad", foreground="#dc2626",
                                           font=("微软雅黑", 10, "bold"))
        self.conclusion_text.tag_configure("highlight", background="#eef2ff",
                                           font=("微软雅黑", 10, "bold"))
        self.conclusion_text.tag_configure("info", foreground="#6b7280")

        self.conclusion_text.configure(state=tk.NORMAL)
        self.conclusion_text.insert(tk.END, "尚无实验数据\n\n", "info")
        self.conclusion_text.insert(tk.END, "完成实验后点击「刷新分析」查看实验结论。")
        self.conclusion_text.configure(state=tk.DISABLED)

    # ---------- 子页4：可视化 ----------
    def _build_visualization_tab(self):
        frame = ttk.Frame(self.page_visual, padding=5)
        frame.pack(fill=tk.BOTH, expand=True)
        card = ttk.LabelFrame(frame, text="训练可视化", padding=5)
        card.pack(fill=tk.BOTH, expand=True)

        self.fig_vis = plt.figure(figsize=(10, 5), dpi=80)
        self.fig_vis.patch.set_facecolor('#ffffff')
        self.ax_vis_loss = self.fig_vis.add_subplot(121)
        self.ax_vis_loss.set_title("训练损失曲线")
        self.ax_vis_loss.set_xlabel("训练轮数 (Epoch)")
        self.ax_vis_loss.set_ylabel("损失值 (Loss)")
        self.ax_vis_loss.grid(True, alpha=0.3)
        self.ax_vis_loss.text(0.5, 0.5, "尚无训练数据", ha='center', va='center',
                              transform=self.ax_vis_loss.transAxes)

        self.ax_vis_err = self.fig_vis.add_subplot(122)
        self.ax_vis_err.set_title("反演误差图 (vp - v_true)")
        self.ax_vis_err.set_xlabel("x / pixel")
        self.ax_vis_err.set_ylabel("y / pixel")
        self.ax_vis_err.text(0.5, 0.5, "尚无反演数据", ha='center', va='center',
                             transform=self.ax_vis_err.transAxes)

        self.canvas_vis = FigureCanvasTkAgg(self.fig_vis, card)
        self.canvas_vis.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_vis.draw()

    # ---------- 子页5：历史记录 ----------
    def _build_history_tab(self):
        paned = ttk.PanedWindow(self.page_history, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        ttk.Label(left_frame, text="实验历史列表", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W, padx=5, pady=2)

        columns = ("#", "时间", "评分")
        self.history_tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.history_tree.heading(col, text=col)
        self.history_tree.column("#", width=35, anchor=tk.CENTER)
        self.history_tree.column("时间", width=160)
        self.history_tree.column("评分", width=70, anchor=tk.CENTER)
        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        ttk.Label(right_frame, text="实验详情", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W, padx=5, pady=2)

        self.history_detail = tk.Text(right_frame, wrap=tk.WORD, state=tk.DISABLED,
                                      font=("微软雅黑", 9), height=20,
                                      bg='#ffffff', fg='#1a1a2e', insertbackground='#1a1a2e',
                                      highlightthickness=1, highlightbackground='#e5e7eb',
                                      relief=tk.FLAT)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.history_detail.yview)
        self.history_detail.configure(yscrollcommand=scrollbar.set)
        self.history_detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._set_text(self.history_detail, "选择左侧历史记录查看详情。")

        btn_frame = ttk.Frame(self.page_history, padding=5)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="🔄 加载选中实验", command=self.app.load_selected_experiment).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📊 比较两个实验", command=self.app.open_compare_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🗑 清空历史", command=self.app.clear_history).pack(side=tk.RIGHT, padx=5)
        ttk.Label(btn_frame, text="提示: 每次反演完成会自动保存快照").pack(side=tk.RIGHT, padx=20)

    # ========== 公共方法（供 app 调用） ==========

    def clear_stats_table(self):
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

    def add_stat_row(self, category, indicator, value, rating):
        self.stats_tree.insert("", tk.END, values=(category, indicator, value, rating))

    def set_patterns_text(self, text):
        """逐行着色：✓绿 ✗红 ~橙 其余灰色"""
        widget = self.patterns_text
        widget.configure(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        for line in text.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("✓"):
                tag = "pass"
            elif stripped.startswith("✗"):
                tag = "fail"
            elif stripped.startswith("~"):
                tag = "warn"
            else:
                tag = "info"
            widget.insert(tk.END, line + "\n", tag)
        widget.configure(state=tk.DISABLED)

    def set_conclusion_text(self, text):
        """富文本渲染：指标蓝色加粗、评级着色、标题高亮"""
        import re
        widget = self.conclusion_text
        widget.configure(state=tk.NORMAL)
        widget.delete(1.0, tk.END)

        metric_re = re.compile(
            r'(RMSE|MAE|损失值|损失|相对误差|动态范围|信噪比|SNR)'
            r'(为|：)?\s*([\d.]+)\s*(m/s|%|s)?'
        )
        quality_re = re.compile(
            r'(优秀|良好|成功(?=实现)|精度高|充裕|清晰|通过)'
            r'|(一般|适中|可接受|基本|模糊)'
            r'|(待提高|不足|较差|偏差|未)'
        )
        highlight_re = re.compile(r'结论[：:]\s*')

        for para in text.split("\n"):
            if not para.strip():
                widget.insert(tk.END, "\n")
                continue
            # highlight marker
            m_hl = highlight_re.match(para)
            if m_hl:
                widget.insert(tk.END, m_hl.group(), "highlight")
                pos = m_hl.end()
            else:
                pos = 0

            # scan remaining text for metrics and quality words
            remaining = para[pos:]
            last = 0
            for m in metric_re.finditer(remaining):
                before = remaining[last:m.start()]
                self._insert_quality_segments(widget, before, quality_re)
                widget.insert(tk.END, m.group(), "metric")
                last = m.end()
            if last < len(remaining):
                self._insert_quality_segments(widget, remaining[last:], quality_re)
            widget.insert(tk.END, "\n")
        widget.configure(state=tk.DISABLED)

    def _insert_quality_segments(self, widget, text, quality_re):
        """在普通文本中查找质量词并交替插入"""
        last = 0
        for m in quality_re.finditer(text):
            widget.insert(tk.END, text[last:m.start()])
            word = m.group()
            if m.group(1):
                tag = "good"
            elif m.group(2):
                tag = "ok"
            else:
                tag = "bad"
            widget.insert(tk.END, word, tag)
            last = m.end()
        if last < len(text):
            widget.insert(tk.END, text[last:])

    def refresh_history_list(self, history):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for snap in history:
            rating_icon = "⭐" if snap['score'] >= 80 else ("✅" if snap['score'] >= 50 else "⚠")
            self.history_tree.insert("", tk.END, iid=str(snap['id']),
                                     values=(snap['id'], snap['timestamp'], f"{rating_icon} {snap['score']}"))

    def update_history_detail(self, text):
        self._set_text(self.history_detail, text)

    def refresh_visualization(self, loss_history, vp, v_true):
        """刷新可视化子页面"""
        self.ax_vis_loss.clear()
        self.ax_vis_err.clear()
        self.ax_vis_loss.tick_params(colors='#6b7280')
        self.ax_vis_err.tick_params(colors='#6b7280')

        if loss_history and len(loss_history) > 0:
            self.ax_vis_loss.plot(range(len(loss_history)), loss_history,
                                  color='#2563eb', linewidth=1.5)
            self.ax_vis_loss.scatter(len(loss_history) - 1, loss_history[-1],
                                     color='#dc2626', s=30, zorder=5,
                                     label=f'终值: {loss_history[-1]:.6f}')
            self.ax_vis_loss.legend()
            self.ax_vis_loss.set_title(f"训练损失曲线 (Epochs={len(loss_history)})")
        else:
            self.ax_vis_loss.text(0.5, 0.5, "尚无训练数据", ha='center', va='center',
                                  transform=self.ax_vis_loss.transAxes)
            self.ax_vis_loss.set_title("训练损失曲线")
        self.ax_vis_loss.set_xlabel("训练轮数 (Epoch)")
        self.ax_vis_loss.set_ylabel("损失值 (Loss)")
        self.ax_vis_loss.grid(True, alpha=0.3)

        if vp is not None and v_true is not None:
            err = vp - v_true
            vmax = max(abs(np.min(err)), abs(np.max(err))) or 1
            im = self.ax_vis_err.imshow(err, cmap='RdBu_r', origin='lower',
                                        vmin=-vmax, vmax=vmax)
            self.fig_vis.colorbar(im, ax=self.ax_vis_err, fraction=0.046, label='m/s')
            rmse = np.sqrt(np.mean(err**2))
            self.ax_vis_err.set_title(f"反演误差  RMSE={rmse:.1f}  [{-vmax:.0f}, {vmax:.0f}] m/s")
        else:
            self.ax_vis_err.text(0.5, 0.5, "尚无反演数据", ha='center', va='center',
                                 transform=self.ax_vis_err.transAxes)
            self.ax_vis_err.set_title("反演误差图")
        self.ax_vis_err.set_xlabel("x / pixel")
        self.ax_vis_err.set_ylabel("y / pixel")
        self.canvas_vis.draw()

    # ---------- 历史选择事件 ----------
    def _on_history_select(self, event):
        selection = self.history_tree.selection()
        if not selection:
            return
        snap_id = int(selection[0])
        snap = next((s for s in self.app.experiment_history if s['id'] == snap_id), None)
        if snap is None:
            return
        lines = [
            f"实验 #{snap['id']}  |  {snap['timestamp']}",
            f"评分: {snap['score']}/100 ({snap['rating']})",
            "-" * 40,
        ]
        if snap['acoustic_params']:
            p = snap['acoustic_params']
            lines.append(f"声波: {p.get('wave_type','?')} {p.get('frequency',0)/1000:.0f} kHz")
        inv = snap['inversion_stats']
        if inv:
            lines.append(f"RMSE: {inv.get('rmse',0):.2f} m/s")
            lines.append(f"MAE: {inv.get('mae',0):.2f} m/s")
            lines.append(f"损失终值: {inv.get('loss_final',0):.6f}")
            lines.append(f"相对误差: {inv.get('relative_error_pct',0):.2f}%")
        lines.append("")
        lines.append("【物理规律检测】")
        for p in snap['patterns']:
            lines.append(f"  {p}")
        lines.append("")
        lines.append("【实验结论】")
        lines.append(snap['conclusion'])
        self._set_text(self.history_detail, "\n".join(lines))

    # ---------- 辅助 ----------
    @staticmethod
    def _set_text(widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)
