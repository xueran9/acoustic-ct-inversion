"""可复用自定义组件：流水线步骤指示器、流水线面板"""
import tkinter as tk
from tkinter import ttk


# ---------- 流水线步骤状态 ----------
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"


class PipelineStep(ttk.Frame):
    """单个流水线步骤指示器组件"""
    STEP_COLORS = {
        PENDING: {"circle": "#d1d5db", "text": "#9ca3af", "label": "#6b7280", "line": "#e5e7eb"},
        RUNNING: {"circle": "#2563eb", "text": "#ffffff", "label": "#2563eb", "line": "#2563eb"},
        COMPLETED: {"circle": "#059669", "text": "#ffffff", "label": "#059669", "line": "#059669"},
        FAILED: {"circle": "#dc2626", "text": "#ffffff", "label": "#dc2626", "line": "#dc2626"},
    }

    def __init__(self, parent, step_num, label, state=PENDING, show_line=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.step_num = step_num
        self.label_text = label
        self._state = state
        self.show_line = show_line
        self.configure(style="Card.TFrame")
        self._build_ui()

    def _build_ui(self):
        self.inner = ttk.Frame(self, style="Card.TFrame")
        self.inner.pack(padx=2, pady=2, fill=tk.X)

        # 圆圈+数字
        self.circle_canvas = tk.Canvas(self.inner, width=36, height=36, highlightthickness=0,
                                       bg='#ffffff', bd=0)
        self.circle_canvas.pack(pady=(0, 2))
        self._draw_circle()

        # 标签
        self.lbl = ttk.Label(self.inner, text=self.label_text, style="Card.TLabel",
                            font=("微软雅黑", 9), anchor=tk.CENTER)
        self.lbl.pack(fill=tk.X)

    def _draw_circle(self):
        self.circle_canvas.delete("all")
        colors = self.STEP_COLORS[self._state]
        cx, cy, r = 18, 18, 13
        self.circle_canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                       fill=colors["circle"], outline=colors["circle"], width=0)
        if self._state == COMPLETED:
            # 勾号
            self.circle_canvas.create_text(cx, cy, text="✓", fill=colors["text"],
                                          font=("微软雅黑", 12, "bold"))
        elif self._state == FAILED:
            self.circle_canvas.create_text(cx, cy, text="✗", fill=colors["text"],
                                          font=("微软雅黑", 12, "bold"))
        else:
            self.circle_canvas.create_text(cx, cy, text=str(self.step_num), fill=colors["text"],
                                          font=("微软雅黑", 10, "bold"))

    def set_state(self, state):
        self._state = state
        self._draw_circle()
        colors = self.STEP_COLORS[state]
        self.lbl.configure(foreground=colors["label"])

    def get_state(self):
        return self._state


class PipelinePanel(ttk.Frame):
    """流水线面板：步骤列表 + 控制按钮"""
    def __init__(self, parent, steps=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.steps = steps or []
        self._step_widgets = []
        self._callbacks = {}
        self._build_ui()

    def set_steps(self, step_labels):
        """设置流水线步骤（标签列表）"""
        self.steps = list(step_labels)
        for w in self._step_widgets:
            w.destroy()
        self._step_widgets = []
        self._build_steps()

    def _build_ui(self):
        # 标题行
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(header, text="实验流水线", style="Header.TLabel").pack(side=tk.LEFT)
        self.status_label = ttk.Label(header, text="就绪", foreground="#6b7280")
        self.status_label.pack(side=tk.RIGHT)

        # 步骤容器
        self.steps_frame = ttk.Frame(self)
        self.steps_frame.pack(fill=tk.X, padx=10, pady=4)
        self._build_steps()

        # 进度条
        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 6))

        # 按钮行
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.btn_run_step = ttk.Button(btn_frame, text="▶ 执行当前步骤",
                                       command=self._on_run_step)
        self.btn_run_step.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_run_all = ttk.Button(btn_frame, text="⏩ 一键运行全部",
                                      style="Accent.TButton", command=self._on_run_all)
        self.btn_run_all.pack(side=tk.LEFT, padx=4)

        self.btn_reset = ttk.Button(btn_frame, text="⟳ 重置", command=self._on_reset)
        self.btn_reset.pack(side=tk.LEFT, padx=4)

    def _build_steps(self):
        if not self.steps:
            ttk.Label(self.steps_frame, text="尚无流水线步骤", foreground="#9ca3af").pack()
            return
        row = ttk.Frame(self.steps_frame)
        row.pack(fill=tk.X)
        for i, label in enumerate(self.steps):
            step = PipelineStep(row, i + 1, label, state=PENDING, show_line=i < len(self.steps) - 1)
            step.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
            self._step_widgets.append(step)

    def set_step_state(self, index, state):
        if 0 <= index < len(self._step_widgets):
            self._step_widgets[index].set_state(state)
            self._update_progress()

    def get_step_state(self, index):
        if 0 <= index < len(self._step_widgets):
            return self._step_widgets[index].get_state()
        return PENDING

    def set_status(self, text):
        self.status_label.configure(text=text)

    def set_on_run_step(self, callback):
        self._callbacks['run_step'] = callback

    def set_on_run_all(self, callback):
        self._callbacks['run_all'] = callback

    def set_on_reset(self, callback):
        self._callbacks['reset'] = callback

    def _on_run_step(self):
        if 'run_step' in self._callbacks:
            self._callbacks['run_step']()

    def _on_run_all(self):
        if 'run_all' in self._callbacks:
            self._callbacks['run_all']()

    def _on_reset(self):
        if 'reset' in self._callbacks:
            self._callbacks['reset']()

    def _update_progress(self):
        if not self._step_widgets:
            self.progress['value'] = 0
            return
        completed = sum(1 for w in self._step_widgets if w.get_state() == COMPLETED)
        failed = sum(1 for w in self._step_widgets if w.get_state() == FAILED)
        done = completed + failed
        pct = int(done / len(self._step_widgets) * 100)
        self.progress['value'] = pct

    def current_step_index(self):
        """返回第一个PENDING或RUNNING的步骤索引"""
        for i, w in enumerate(self._step_widgets):
            state = w.get_state()
            if state in (PENDING, RUNNING):
                return i
        return -1  # all done

    def reset_all(self):
        """将所有步骤重置为PENDING"""
        for w in self._step_widgets:
            w.set_state(PENDING)
        self.progress['value'] = 0
        self.set_status("就绪")
