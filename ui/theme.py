"""浅色主题配置 — 白色底色，清晰色彩层级"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import subprocess


def _find_chinese_font():
    """Windows 下查找可用的中文字体"""
    if os.name != 'nt':
        return 'Arial'
    candidates = [
        'Microsoft YaHei', 'Microsoft YaHei UI',
        'SimHei', 'SimSun', 'NSimSun',
        'FangSong', 'KaiTi',
    ]
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    # 手动搜索 fonts 目录
    fonts_dir = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
    for fname in ['msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'simsun.ttc']:
        fpath = os.path.join(fonts_dir, fname)
        if os.path.exists(fpath):
            try:
                fm.fontManager.addfont(fpath)
                fp = FontProperties(fname=fpath)
                name = fp.get_name()
                if name:
                    return name
                return fname.split('.')[0]
            except Exception:
                continue
    from matplotlib import font_manager
    font_manager._load_fontmanager(try_read_cache=False)
    return 'Arial'


# 中文支持
_CN_FONT = _find_chinese_font()


# ===================== 浅色配色方案 =====================
LIGHT_THEME = {
    # 背景层级
    'bg_root': '#f5f6fa',       # Lv0 窗口/页面最底层
    'bg_page': '#ffffff',       # Lv1 内容页面/面板
    'bg_card': '#ffffff',       # 卡片主体（白底）
    'bg_card_header': '#f8f9fc', # 卡片标题区
    'bg_input': '#f0f2f5',      # 输入框背景
    'bg_hover': '#eef1f8',      # 悬停高亮
    # 文字颜色
    'fg_primary': '#1a1a2e',    # 主文字（深色）
    'fg_secondary': '#6b7280',  # 次要/说明文字
    'fg_disabled': '#b0b7c3',   # 禁用文字
    # 强调色
    'accent_blue': '#2563eb',   # 按钮/链接/选中
    'accent_blue_hover': '#1d4ed8',
    'accent_green': '#059669',  # 成功/进度
    'accent_green_hover': '#047857',
    'accent_orange': '#d97706', # 警告
    'accent_red': '#dc2626',    # 错误
    'accent_red_hover': '#b91c1c',
    # 边框与分隔
    'border': '#e5e7eb',        # 卡片/容器分隔线
    'border_focus': '#2563eb',  # 输入框聚焦边框
    # 阴影（通过边框模拟）
    'shadow': '#d1d5db',
}


def apply_light_theme(root):
    """应用浅色主题到整个应用"""
    T = LIGHT_THEME

    # === Matplotlib 全局配色（含中文字体） ===
    plt.rcParams.update({
        'font.family': [_CN_FONT, 'Arial'],
        'axes.unicode_minus': False,
        'axes.facecolor': '#ffffff',
        'axes.edgecolor': '#e5e7eb',
        'axes.labelcolor': '#6b7280',
        'axes.titlecolor': '#1a1a2e',
        'xtick.color': '#6b7280',
        'ytick.color': '#6b7280',
        'grid.color': '#e5e7eb',
        'grid.alpha': 0.5,
        'figure.facecolor': '#f5f6fa',
        'savefig.facecolor': '#f5f6fa',
    })
    plt.style.use('default')

    # === ttk.Style ===
    style = ttk.Style()
    style.theme_use('clam')

    # 全局默认
    style.configure(".", background=T['bg_root'], foreground=T['fg_primary'],
                    fieldbackground=T['bg_input'], selectbackground=T['accent_blue'],
                    selectforeground='#ffffff')

    # Frame
    style.configure("TFrame", background=T['bg_root'])
    style.configure("Card.TFrame", background=T['bg_card'], relief=tk.RIDGE, borderwidth=1)

    # Label
    style.configure("TLabel", background=T['bg_root'], foreground=T['fg_primary'], font=("微软雅黑", 10))
    style.configure("Card.TLabel", background=T['bg_card'], foreground=T['fg_primary'])
    style.configure("Header.TLabel", background=T['bg_root'], foreground=T['fg_primary'],
                    font=("微软雅黑", 12, "bold"))
    style.configure("Title.TLabel", background=T['bg_root'], foreground=T['fg_primary'],
                    font=("微软雅黑", 14, "bold"))

    # Button
    style.configure("TButton", background=T['bg_page'], foreground=T['fg_primary'],
                    bordercolor=T['border'], font=("微软雅黑", 10), padding=[10, 4])
    style.map("TButton", background=[("active", T['bg_hover']), ("pressed", T['border'])])

    # Accent button (blue solid)
    style.configure("Accent.TButton", background=T['accent_blue'], foreground='#ffffff',
                    bordercolor=T['accent_blue'], font=("微软雅黑", 10, "bold"), padding=[10, 6])
    style.map("Accent.TButton", background=[("active", T['accent_blue_hover']), ("pressed", T['accent_blue_hover'])])

    # Green button (success)
    style.configure("Success.TButton", background=T['accent_green'], foreground='#ffffff',
                    bordercolor=T['accent_green'], font=("微软雅黑", 10, "bold"), padding=[10, 6])
    style.map("Success.TButton", background=[("active", T['accent_green_hover'])])

    # Red button (danger)
    style.configure("Danger.TButton", background=T['accent_red'], foreground='#ffffff',
                    bordercolor=T['accent_red'], font=("微软雅黑", 10), padding=[10, 4])
    style.map("Danger.TButton", background=[("active", T['accent_red_hover'])])

    # Entry
    style.configure("TEntry", fieldbackground=T['bg_input'], foreground=T['fg_primary'],
                    insertcolor=T['fg_primary'], bordercolor=T['border'], font=("微软雅黑", 10))
    style.map("TEntry", bordercolor=[("focus", T['border_focus'])])

    # Combobox
    style.configure("TCombobox", fieldbackground=T['bg_input'], foreground=T['fg_primary'],
                    arrowcolor=T['fg_secondary'], bordercolor=T['border'], font=("微软雅黑", 10))
    style.map("TCombobox", bordercolor=[("focus", T['border_focus'])])

    # Spinbox
    style.configure("TSpinbox", fieldbackground=T['bg_input'], foreground=T['fg_primary'], bordercolor=T['border'])

    # Scale
    style.configure("Horizontal.TScale", background=T['bg_root'], troughcolor=T['bg_input'])

    # Notebook
    style.configure("TNotebook", background=T['bg_root'], tabmargins=[2, 5, 2, 0], borderwidth=0)
    style.configure("TNotebook.Tab", background=T['border'], foreground=T['fg_secondary'],
                    padding=[12, 4], font=("微软雅黑", 10))
    style.map("TNotebook.Tab", background=[("selected", T['bg_page'])],
              foreground=[("selected", T['accent_blue'])])

    # Treeview
    style.configure("Treeview", background=T['bg_page'], foreground=T['fg_primary'],
                    fieldbackground=T['bg_page'], rowheight=30, font=("微软雅黑", 10),
                    bordercolor=T['border'], borderwidth=1)
    style.configure("Treeview.Heading", background=T['bg_card_header'], foreground=T['fg_primary'],
                    font=("微软雅黑", 10, "bold"), bordercolor=T['border'])
    style.map("Treeview", background=[("selected", T['accent_blue'])],
              foreground=[("selected", '#ffffff')])

    # LabelFrame
    style.configure("TLabelframe", background=T['bg_card'], foreground=T['fg_primary'],
                    bordercolor=T['border'], relief=tk.RIDGE, borderwidth=1)
    style.configure("TLabelframe.Label", background=T['bg_card'], foreground=T['accent_blue'],
                    font=("微软雅黑", 10, "bold"))

    # Progressbar
    style.configure("Horizontal.TProgressbar", background=T['accent_blue'], troughcolor=T['bg_input'])
    style.configure("green.Horizontal.TProgressbar", background=T['accent_green'], troughcolor=T['bg_input'])

    # Scrollbar
    style.configure("Vertical.TScrollbar", background=T['bg_page'], troughcolor=T['bg_root'],
                    arrowcolor=T['fg_secondary'])
    style.configure("Horizontal.TScrollbar", background=T['bg_page'], troughcolor=T['bg_root'],
                    arrowcolor=T['fg_secondary'])

    # PanedWindow
    style.configure("TPanedwindow", background=T['border'])

    # Separator
    style.configure("TSeparator", background=T['border'])

    # 全局字体
    root.option_add("*font", ("微软雅黑", 10))
    root.option_add("*TCombobox*Listbox.background", T['bg_page'])
    root.option_add("*TCombobox*Listbox.foreground", T['fg_primary'])
    root.option_add("*TCombobox*Listbox.selectBackground", T['accent_blue'])
    root.option_add("*TCombobox*Listbox.selectForeground", '#ffffff')

    # 根窗口背景
    root.configure(bg=T['bg_root'])

    return T  # 返回主题字典供其他模块使用
