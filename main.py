#!/usr/bin/env python3
"""AI驱动声学CT反演实验系统 — 多模块版本入口"""
import sys
import os

# 确保当前目录为项目根目录（支持桌面双击启动）
if '__main__' == __name__:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import ActCTApp

if __name__ == '__main__':
    app = ActCTApp()
    app.mainloop()
