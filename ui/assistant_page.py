"""页面4：AI 实验助手 — 对话界面 + 快捷按钮"""
import tkinter as tk
from tkinter import ttk


class AssistantPage(ttk.Frame):
    """AI实验助手页面 — 聊天交互"""
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        # ========== 聊天显示区 ==========
        frame_chat = ttk.LabelFrame(self, text="对话记录", padding=5)
        frame_chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self.chat_text = tk.Text(frame_chat, wrap=tk.WORD, state=tk.DISABLED,
                                  font=("微软雅黑", 10), bg='#ffffff', fg='#1a1a2e',
                                  insertbackground='#1a1a2e', height=25,
                                  highlightthickness=1, highlightbackground='#e5e7eb',
                                  relief=tk.FLAT)
        scrollbar = ttk.Scrollbar(frame_chat, orient=tk.VERTICAL, command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=scrollbar.set)
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 彩色标签
        self.chat_text.tag_configure("user_prefix", foreground="#2563eb",
                                      font=("微软雅黑", 10, "bold"))
        self.chat_text.tag_configure("assistant_prefix", foreground="#059669",
                                      font=("微软雅黑", 10, "bold"))
        self.chat_text.tag_configure("system_prefix", foreground="#6b7280",
                                      font=("微软雅黑", 10, "bold"))
        self.chat_text.tag_configure("separator", foreground="#e5e7eb")
        self.chat_text.tag_configure("user_text", foreground="#1a1a2e")
        self.chat_text.tag_configure("assistant_text", foreground="#1a1a2e")

        self._update_display()

        # ========== 输入区 ==========
        frame_input = ttk.LabelFrame(self, text="输入", padding=8)
        frame_input.pack(fill=tk.X, padx=10, pady=(5, 0))

        input_row = ttk.Frame(frame_input)
        input_row.pack(fill=tk.X)
        self.chat_entry = ttk.Entry(input_row, font=("微软雅黑", 10))
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.chat_entry.bind("<Return>", self._on_send)
        ttk.Button(input_row, text="发送", command=self._on_send).pack(side=tk.RIGHT)

        # ========== 快捷按钮区 ==========
        frame_btns = ttk.Frame(self, padding=8)
        frame_btns.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(frame_btns, text="✅ 验证数据", command=self._quick_validate).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btns, text="⭐ 质量评分", command=self._quick_quality).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btns, text="💡 改进建议", command=self._quick_suggestions).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btns, text="📋 完整报告", command=self._quick_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btns, text="🗑 清空对话", command=self._quick_clear).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_btns, text="📄 导出报告", command=self.app.export_txt_report).pack(side=tk.LEFT, padx=2)

    def _on_send(self, event=None):
        text = self.chat_entry.get().strip()
        if not text:
            return
        self.chat_entry.delete(0, tk.END)
        self.app.assistant.process_query(text)
        self._update_display()

    def _quick_validate(self):
        self.app.assistant.process_query("验证数据合理性")
        self._update_display()

    def _quick_quality(self):
        self.app.assistant.process_query("实验质量评分")
        self._update_display()

    def _quick_suggestions(self):
        self.app.assistant.process_query("改进建议")
        self._update_display()

    def _quick_report(self):
        self.app.assistant.process_query("生成完整报告")
        self._update_display()

    def _quick_clear(self):
        self.app.assistant.clear_chat()
        self._update_display()

    def _update_display(self):
        """刷新聊天显示区"""
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete(1.0, tk.END)

        history = self.app.assistant.chat_history
        if not history:
            welcome = self.app.assistant.get_welcome_message()
            self.chat_text.insert(tk.END, welcome)
        else:
            for speaker, text in history:
                if speaker == "用户":
                    self.chat_text.insert(tk.END, "[用户] ", "user_prefix")
                    self.chat_text.insert(tk.END, text + "\n", "user_text")
                elif speaker == "助手":
                    self.chat_text.insert(tk.END, "[助手] ", "assistant_prefix")
                    self.chat_text.insert(tk.END, text + "\n", "assistant_text")
                elif speaker == "系统":
                    self.chat_text.insert(tk.END, "[系统] ", "system_prefix")
                    self.chat_text.insert(tk.END, text + "\n")
                self.chat_text.insert(tk.END, "-" * 50 + "\n", "separator")

        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def refresh_display(self):
        """Public method called by app to refresh chat"""
        self._update_display()
