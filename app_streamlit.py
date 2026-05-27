"""声波CT反演实验系统 - Streamlit Web版本"""
import sys, os, io, json, time, re
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter

import streamlit as st

# 项目模块
from core.physics import (gen_phantom, gen_phantom_simple, gen_phantom_complex,
                            gen_phantom_layered, gen_phantom_tumor, gen_phantom_custom, forward_proj)
from core.model import Net
from core.analyzer import DataAnalyzer
from core.assistant import AIAssistant
from core.collector import WaveCollector
from core.wave_generator import generate_acoustic_wave
from core.lab_equipment import (WAVE_SOURCES, SENSOR_ARRAYS, PHANTOM_TYPES,
                                compute_equipment_effects, get_equipment_summary, build_acoustic_params)


# ====================================================================
# 页面配置
# ====================================================================
st.set_page_config(
    page_title="AI声波CT反演实验系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================
# Matplotlib 中文字体
# ====================================================================
def _find_chinese_font():
    for f in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'sans-serif']:
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            return f
        except Exception:
            continue
    return 'sans-serif'

_chinese_font = _find_chinese_font()
plt.rcParams['font.family'] = _chinese_font
plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# CSS自定义样式
# ====================================================================
st.markdown("""
<style>
    .stButton > button { border-radius: 6px; font-weight: 500; }
    .step-card {
        background: #f8fafc; border-radius: 10px; padding: 16px;
        border-left: 4px solid #2563eb; margin: 6px 0;
    }
    .step-card.completed { border-left-color: #059669; }
    .step-card.running { border-left-color: #2563eb; }
    .step-card.failed { border-left-color: #dc2626; }
    .conclusion-box {
        background: #f8fafc; border-left: 4px solid #2563eb;
        border-radius: 8px; padding: 20px; margin: 10px 0;
    }
    .patterns-box {
        background: #f8fafc; border-left: 4px solid #2563eb;
        border-radius: 8px; padding: 20px; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# Session State 初始化
# ====================================================================
def init_session_state():
    if "initialized" in st.session_state:
        return

    st.session_state.size = 64
    st.session_state.net = None
    st.session_state.v_true = None
    st.session_state.sino = None
    st.session_state.vp = None
    st.session_state.loss_history = []

    st.session_state.wave_freq_val = 50000.0
    st.session_state.wave_amp_val = 1.0
    st.session_state.wave_type_val = "正弦波"
    st.session_state.external_wave = None
    st.session_state.wav_filename = None

    st.session_state.analyzer = DataAnalyzer()
    st.session_state.assistant = AIAssistant(st.session_state.analyzer)
    st.session_state.wave_collector = WaveCollector()

    st.session_state.experiment_history = []
    st.session_state.exp_counter = 0

    st.session_state.pipeline_steps = [
        {"name": "生成声场", "state": "pending"},
        {"name": "声波投影", "state": "pending"},
        {"name": "AI反演",   "state": "pending"},
        {"name": "数据分析", "state": "pending"},
        {"name": "保存快照", "state": "pending"},
    ]

    st.session_state.status_message = "就绪，等待操作"
    st.session_state.chat_history_msgs = []
    st.session_state.history_select_id = None
    st.session_state.compare_ids = []

    st.session_state.initialized = True

init_session_state()

# ====================================================================
# 辅助函数
# ====================================================================
def step_icon(state):
    return {"pending": "⚪", "running": "🔵", "completed": "🟢", "failed": "🔴"}.get(state, "⚪")

def reset_pipeline():
    for s in st.session_state.pipeline_steps:
        s["state"] = "pending"
    st.session_state.v_true = None
    st.session_state.sino = None
    st.session_state.vp = None
    st.session_state.net = None
    st.session_state.loss_history = []
    st.session_state.analyzer.clear()
    st.session_state.assistant = AIAssistant(st.session_state.analyzer)
    st.session_state.status_message = "流水线已重置"

def get_acoustic_params():
    return {
        'frequency': st.session_state.wave_freq_val,
        'amplitude': st.session_state.wave_amp_val,
        'wave_type': st.session_state.wave_type_val,
        'external': st.session_state.external_wave,
    }

# ---- Matplotlib 图表工厂 ----
def make_fig_phantom(v_true):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(v_true, cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
    ax.set_title("真实声速场", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, label='m/s')
    fig.tight_layout()
    return fig

def make_fig_sino(sino):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sino, cmap='gray', aspect='auto', extent=[0, 180, -1, 1])
    ax.set_title("声波投影Sinogram", fontsize=11)
    ax.set_xlabel("角度 (°)"); ax.set_ylabel("探测器位置")
    fig.tight_layout()
    return fig

def make_fig_inversion(vp):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(vp, cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
    ax.set_title("AI反演结果", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, label='m/s')
    fig.tight_layout()
    return fig

def make_fig_loss(loss_history):
    fig, ax = plt.subplots(figsize=(4, 3))
    if loss_history:
        ax.plot(range(len(loss_history)), loss_history, color='#2563eb', linewidth=1.5)
        ax.scatter(len(loss_history)-1, loss_history[-1], color='#dc2626', s=30, zorder=5,
                   label=f'终值: {loss_history[-1]:.6f}')
        ax.legend(fontsize=8)
    ax.set_title(f"训练损失曲线 (Epochs={len(loss_history)})", fontsize=10)
    ax.set_xlabel("训练轮数"); ax.set_ylabel("损失值")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def make_fig_error(vp, v_true):
    fig, ax = plt.subplots(figsize=(4, 3))
    err = vp - v_true
    vmax = max(abs(np.min(err)), abs(np.max(err))) or 1
    im = ax.imshow(err, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, label='m/s')
    rmse = np.sqrt(np.mean(err**2))
    ax.set_title(f"反演误差  RMSE={rmse:.1f} m/s", fontsize=10)
    ax.set_xlabel("x / pixel"); ax.set_ylabel("y / pixel")
    fig.tight_layout()
    return fig

def make_fig_waveform(t, wave, title="波形预览"):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(t[:500], wave[:500], color='#2563eb', linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("振幅")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ---- 流水线步骤函数 ----
def step0_generate_phantom():
    step = st.session_state.pipeline_steps[0]
    step["state"] = "running"
    try:
        st.session_state.v_true = gen_phantom(st.session_state.size)
        st.session_state.analyzer.set_phantom(st.session_state.v_true)
        step["state"] = "completed"
        st.session_state.status_message = "声场已生成 (64x64, 含高速/低速异常区)"
    except Exception as e:
        step["state"] = "failed"
        st.session_state.status_message = f"生成声场失败: {e}"

def step1_forward_projection():
    if st.session_state.v_true is None:
        st.warning("请先生成声场")
        return
    step = st.session_state.pipeline_steps[1]
    step["state"] = "running"
    try:
        params = get_acoustic_params()
        st.session_state.sino = forward_proj(st.session_state.v_true, acoustic_params=params)
        st.session_state.analyzer.set_sino(st.session_state.sino, params)
        step["state"] = "completed"
        st.session_state.status_message = "声波投影已完成 (90角度 x 128探测器)"
    except Exception as e:
        step["state"] = "failed"
        st.session_state.status_message = f"投影失败: {e}"

def step2_ai_inversion():
    if st.session_state.sino is None:
        st.warning("请先生成投影数据")
        return
    step = st.session_state.pipeline_steps[2]
    step["state"] = "running"
    st.session_state.status_message = "AI反演训练中..."

    net = Net(st.session_state.size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    st_tensor = torch.FloatTensor(st.session_state.sino).unsqueeze(0)
    vt_tensor = torch.FloatTensor(st.session_state.v_true).unsqueeze(0)

    loss_history = []
    epochs = 200

    pb = st.progress(0)
    status_text = st.empty()
    loss_placeholder = st.empty()

    for ep in range(epochs):
        optimizer.zero_grad()
        output = net(st_tensor)
        loss = criterion(output, vt_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        pb.progress((ep + 1) / epochs)
        if ep % 10 == 0:
            status_text.text(f"训练中... Epoch {ep}/{epochs}  Loss: {loss.item():.6f}")

    pb.progress(1.0)
    status_text.text(f"训练完成！最终损失: {loss_history[-1]:.6f}")

    st.session_state.net = net
    st.session_state.vp = net(st_tensor).detach().numpy().squeeze()
    st.session_state.loss_history = loss_history
    st.session_state.analyzer.set_inversion(st.session_state.vp, loss_history)
    step["state"] = "completed"
    st.session_state.status_message = f"AI反演完成 (200轮, 终损={loss_history[-1]:.6f})"

def step3_analyze():
    step = st.session_state.pipeline_steps[3]
    step["state"] = "running"
    try:
        st.session_state.analyzer.compute_all_stats()
        st.session_state.analyzer.detect_patterns()
        st.session_state.analyzer.generate_conclusion()
        st.session_state.assistant.refresh()
        step["state"] = "completed"
        st.session_state.status_message = "数据分析完成"
    except Exception as e:
        step["state"] = "failed"
        st.session_state.status_message = f"分析失败: {e}"

def step4_save_snapshot():
    step = st.session_state.pipeline_steps[4]
    step["state"] = "running"
    try:
        analyzer = st.session_state.analyzer
        assistant = st.session_state.assistant
        report = analyzer.get_report_dict()

        st.session_state.exp_counter += 1
        snap = {
            'id': st.session_state.exp_counter,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'score': assistant.experiment_score or 0,
            'rating': assistant.quality_rating or '未评分',
            'acoustic_params': dict(get_acoustic_params()),
            'v_true': st.session_state.v_true.tolist() if st.session_state.v_true is not None else None,
            'sino': st.session_state.sino.tolist() if st.session_state.sino is not None else None,
            'vp': st.session_state.vp.tolist() if st.session_state.vp is not None else None,
            'loss_history': list(st.session_state.loss_history),
            'inversion_stats': dict(report.get('stats', {})),
            'patterns': list(report.get('patterns', [])),
            'conclusion': report.get('conclusion', ''),
        }
        st.session_state.experiment_history.insert(0, snap)
        if len(st.session_state.experiment_history) > 50:
            st.session_state.experiment_history = st.session_state.experiment_history[:50]
        step["state"] = "completed"
        st.session_state.status_message = f"快照 #{snap['id']} 已保存"
    except Exception as e:
        step["state"] = "failed"
        st.session_state.status_message = f"保存失败: {e}"

# ---- 历史持久化 ----
def persist_history():
    d = Path(__file__).parent / ".streamlit_history"
    d.mkdir(parents=True, exist_ok=True)
    serializable = []
    for snap in st.session_state.experiment_history:
        s = dict(snap)
        for k in ('v_true', 'sino', 'vp'):
            if s.get(k) is not None:
                s[k] = s[k] if isinstance(s[k], list) else s[k].tolist() if hasattr(s[k], 'tolist') else s[k]
        serializable.append(s)
    fp = d / "experiments.json"
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    st.sidebar.success(f"已保存 {len(serializable)} 条记录")

def load_history():
    fp = Path(__file__).parent / ".streamlit_history" / "experiments.json"
    if not fp.exists():
        st.sidebar.info("没有找到历史记录")
        return
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for snap in data:
        for k in ('v_true', 'sino', 'vp'):
            if snap.get(k) is not None:
                snap[k] = np.array(snap[k])
    st.session_state.experiment_history = data
    st.session_state.exp_counter = max((s['id'] for s in data), default=0)
    st.sidebar.success(f"已加载 {len(data)} 条记录")

# ---- 导出辅助 ----
def build_txt_bytes():
    analyzer = st.session_state.analyzer
    assistant = st.session_state.assistant
    assistant.refresh()
    report = analyzer.get_report_dict()
    stats = report['stats']
    lines = ["=" * 50, "  声波CT反演实验报告", "=" * 50, ""]
    lines.append("【实验参数】")
    p = get_acoustic_params()
    lines.append(f"  波形类型: {p.get('wave_type','未知')}")
    lines.append(f"  频率: {p.get('frequency',0)/1000:.1f} kHz")
    lines.append(f"  振幅: {p.get('amplitude',1.0):.2f}")
    lines.append("")
    if analyzer.has_phantom():
        lines.append("【声速场统计】")
        for k, label in [('v_min','最小值'),('v_max','最大值'),('v_mean','均值'),('v_std','标准差'),('grad_mean','梯度均值')]:
            lines.append(f"  {label}: {stats.get(k,0):.2f}")
        lines.append(f"  异常区域占比: {stats.get('anomaly_pct',0):.1f}%")
        lines.append("")
    if analyzer.has_sino():
        lines.append("【投影数据统计】")
        lines.append(f"  动态范围: {stats.get('dynamic_range',0):.6f} s")
        lines.append(f"  信噪比: {stats.get('snr_estimate',0):.2f}")
        lines.append(f"  角度一致性: {stats.get('mean_cross_corr',0):.4f}")
        lines.append("")
    if analyzer.has_inversion():
        lines.append("【反演结果统计】")
        lines.append(f"  RMSE: {stats.get('rmse',0):.2f} m/s")
        lines.append(f"  MAE: {stats.get('mae',0):.2f} m/s")
        lines.append(f"  相对误差: {stats.get('relative_error_pct',0):.2f}%")
        lines.append(f"  损失终值: {stats.get('loss_final',0):.6f}")
        lines.append("")
    lines.append("【物理规律检测】")
    for pat in report['patterns']:
        lines.append(f"  {pat}")
    lines.append("")
    lines.append("【AI评估】")
    lines.append(f"  评分: {assistant.experiment_score or 0}/100 — {assistant.quality_rating or '无数据'}")
    lines.append("")
    lines.append("【实验结论】")
    lines.append(f"  {report['conclusion']}")
    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines).encode('utf-8')

def build_csv_bytes():
    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["类别", "指标", "数值", "评价"])
    for row in st.session_state.analyzer.get_csv_rows():
        writer.writerow(row)
    return buf.getvalue().encode('utf-8-sig')


# ====================================================================
# 侧边栏
# ====================================================================
with st.sidebar:
    st.markdown("## 🔬 声学CT反演实验系统")
    st.markdown("---")

    st.markdown("### 🔈 声波参数")
    freq_khz = st.number_input("频率 (kHz)", min_value=1.0, max_value=200.0, value=st.session_state.wave_freq_val/1000, step=1.0)
    st.session_state.wave_freq_val = freq_khz * 1000
    st.session_state.wave_amp_val = st.number_input("振幅", min_value=0.1, max_value=10.0, value=float(st.session_state.wave_amp_val), step=0.1)
    st.session_state.wave_type_val = st.selectbox("波形类型", ["正弦波", "方波", "脉冲波"],
                                                   index=["正弦波", "方波", "脉冲波"].index(st.session_state.wave_type_val) if st.session_state.wave_type_val in ["正弦波", "方波", "脉冲波"] else 0)

    # WAV文件上传
    with st.expander("📁 加载WAV文件"):
        uploaded = st.file_uploader("选择WAV文件", type=["wav"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                sr, data = wavfile.read(io.BytesIO(uploaded.read()))
                if data.ndim > 1:
                    data = data[:, 0]
                mx = np.max(np.abs(data)) or 1
                st.session_state.external_wave = data.astype(np.float64) / mx
                st.session_state.wav_filename = uploaded.name
                st.success(f"已加载: {uploaded.name} ({sr} Hz, {len(data)} samples)")
            except Exception as e:
                st.error(f"读取失败: {e}")

    # 波形预览
    if st.button("📈 预览波形"):
        params = get_acoustic_params()
        if st.session_state.external_wave is not None:
            wave = st.session_state.external_wave
            t = np.arange(len(wave)) / 44100
            title = f"WAV: {st.session_state.wav_filename or 'external'}"
        else:
            t, wave = generate_acoustic_wave(params['frequency'], params['amplitude'], 0.004, params['wave_type'])
            title = f"{params['wave_type']} {params['frequency']/1000:.0f} kHz"
        fig = make_fig_waveform(t, wave, title)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("### ⚙ 流水线控制")
    if st.button("🔄 重置流水线", use_container_width=True):
        reset_pipeline()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📤 导出报告")
    has_data = st.session_state.analyzer.has_phantom() or st.session_state.analyzer.has_sino() or st.session_state.analyzer.has_inversion()
    st.download_button("📄 下载TXT报告", data=build_txt_bytes() if has_data else b"",
                       file_name="CT_report.txt", disabled=not has_data, use_container_width=True)
    st.download_button("📊 下载CSV数据", data=build_csv_bytes() if has_data else b"",
                       file_name="CT_stats.csv", disabled=not has_data, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💾 历史记录")
    c1, c2 = st.columns(2)
    if c1.button("💾 保存", use_container_width=True):
        persist_history()
    if c2.button("📂 加载", use_container_width=True):
        load_history()
        st.rerun()


# ====================================================================
# 主页面 — 4个标签页
# ====================================================================
st.title("🔬 AI驱动声波CT反演实验系统")
tabs = st.tabs(["CT反演实验", "声波输入口", "实验数据分析", "AI实验助手", "虚拟实验室"])

# ================================
# Tab 1: CT反演实验
# ================================
with tabs[0]:
    # 步骤指示器
    steps = st.session_state.pipeline_steps
    cols = st.columns(5)
    for i, (col, s) in enumerate(zip(cols, steps)):
        with col:
            icon = step_icon(s["state"])
            st.markdown(f"**{icon} {s['name']}**")
            if s["state"] == "completed":
                st.caption("✅ 完成")
            elif s["state"] == "running":
                st.caption("⏳ 执行中...")
            elif s["state"] == "failed":
                st.caption("❌ 失败")
            else:
                st.caption("—")

    # 总体进度
    done = sum(1 for s in steps if s["state"] == "completed")
    st.progress(done / 5, text=f"进度: {done}/5 已完成")

    # 状态信息
    st.info(st.session_state.status_message)

    # 操作按钮
    btn_cols = st.columns([1, 1, 3])
    with btn_cols[0]:
        # 找到第一个pending步骤
        next_idx = next((i for i, s in enumerate(steps) if s["state"] in ("pending", "failed")), None)
        if next_idx is not None:
            label = f"▶ 执行: {steps[next_idx]['name']}"
            if st.button(label, use_container_width=True, type="primary"):
                funcs = [step0_generate_phantom, step1_forward_projection,
                         step2_ai_inversion, step3_analyze, step4_save_snapshot]
                funcs[next_idx]()
                st.rerun()
        else:
            st.button("✅ 全部完成", disabled=True, use_container_width=True)

    with btn_cols[1]:
        all_done = all(s["state"] == "completed" for s in steps)
        if st.button("🚀 一键执行全部", use_container_width=True, disabled=all_done):
            funcs = [step0_generate_phantom, step1_forward_projection,
                     step2_ai_inversion, step3_analyze, step4_save_snapshot]
            for i, fn in enumerate(funcs):
                if steps[i]["state"] in ("pending", "failed"):
                    fn()
            st.rerun()

    st.markdown("---")

    # 3列图表
    if st.session_state.v_true is not None or st.session_state.sino is not None or st.session_state.vp is not None:
        plot_cols = st.columns(3)
        with plot_cols[0]:
            if st.session_state.v_true is not None:
                fig = make_fig_phantom(st.session_state.v_true)
                st.pyplot(fig); plt.close(fig)
            else:
                st.info("尚未生成声场")
        with plot_cols[1]:
            if st.session_state.sino is not None:
                fig = make_fig_sino(st.session_state.sino)
                st.pyplot(fig); plt.close(fig)
            else:
                st.info("尚未生成投影")
        with plot_cols[2]:
            if st.session_state.vp is not None:
                fig = make_fig_inversion(st.session_state.vp)
                st.pyplot(fig); plt.close(fig)
            else:
                st.info("尚未执行反演")


# ================================
# Tab 2: 声波输入口
# ================================
with tabs[1]:
    collector = st.session_state.wave_collector

    # 采集状态
    stcols = st.columns(4)
    stcols[0].metric("状态", "采集中" if collector.is_collecting else ("已完成" if collector.collection_complete else "未开始"))
    stcols[1].metric("目标", str(collector.target_count))
    stcols[2].metric("已接受", str(len(collector.accepted_waves)))
    stcols[3].metric("已拒绝", str(len(collector.rejected_waves)))

    # 进度条
    if collector.is_collecting or collector.collection_complete:
        progress = len(collector.accepted_waves) / max(collector.target_count, 1)
        st.progress(min(progress, 1.0), text=f"采集进度 {len(collector.accepted_waves)}/{collector.target_count}")

    # 控制按钮
    ctl_cols = st.columns(3)
    if ctl_cols[0].button("▶ 开始采集", use_container_width=True, disabled=collector.is_collecting):
        collector.start_collection(5)
        st.rerun()
    if ctl_cols[1].button("⏹ 提前结束", use_container_width=True, disabled=not collector.is_collecting):
        collector.stop_early()
        st.rerun()
    if ctl_cols[2].button("🗑 清除", use_container_width=True):
        collector.clear()
        st.rerun()

    st.markdown("---")

    # 提交声波
    st.markdown("### 📡 提交声波样本")
    if collector.is_collecting and not collector.collection_complete:
        submit_cols = st.columns([2, 1])
        with submit_cols[0]:
            if st.button("📤 提交当前参数声波", use_container_width=True, type="primary"):
                params = get_acoustic_params()
                if st.session_state.external_wave is not None:
                    wave = st.session_state.external_wave
                else:
                    t, wave = generate_acoustic_wave(params['frequency'], params['amplitude'], 0.004, params['wave_type'])
                result = collector.submit_wave(wave, params)
                if result['accepted']:
                    st.success(f"接受: {result['reason']}")
                else:
                    if result.get('type_mismatch'):
                        st.warning(f"类型不匹配: {result['reason']}")
                        if st.button("强制提交(忽略类型检查)", key="force_submit"):
                            r2 = collector.submit_wave(wave, params, override=True)
                            if r2['accepted']:
                                st.success("已强制接受")
                            else:
                                st.error(r2['reason'])
                            st.rerun()
                    else:
                        st.error(f"拒绝: {result['reason']}")
                if result.get('complete'):
                    st.balloons()
                    st.success("🎉 采集完成！")
                st.rerun()
    elif collector.collection_complete:
        st.success(f"✅ 采集已完成，共接受 {len(collector.accepted_waves)} 次声波。")
        if st.button("📊 查看统一参数"):
            unified = collector.unify_waves()
            if unified:
                st.json(unified)
    else:
        st.info("请先点击「开始采集」启动会话。")

    st.markdown("---")

    # 采集日志
    st.markdown("### 📋 采集日志")
    st.code(collector.get_log_text() or "尚无采集记录。", language=None)

    # 波形预览
    if collector.accepted_waves:
        st.markdown("### 📈 已接受波形预览")
        fig, ax = plt.subplots(figsize=(8, 3))
        for w in collector.accepted_waves:
            wd = w['wave_data']
            t = np.arange(len(wd)) / 44100 if st.session_state.external_wave is not None else np.linspace(0, 0.004, len(wd))
            ax.plot(t[:500], wd[:500], alpha=0.6, linewidth=0.8,
                    label=f"#{w['index']} {w['acoustic_params'].get('wave_type','?')}")
        ax.set_title("已接受声波叠加"); ax.set_xlabel("时间 (s)"); ax.set_ylabel("振幅")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)


# ================================
# Tab 3: 实验数据分析
# ================================
with tabs[2]:
    analyzer = st.session_state.analyzer
    has_data = analyzer.has_phantom() or analyzer.has_sino() or analyzer.has_inversion()

    # 工具栏
    top_cols = st.columns([1, 1, 4])
    if top_cols[0].button("🔄 刷新分析", use_container_width=True):
        if has_data:
            step3_analyze()
            st.rerun()
    if top_cols[1].button("💾 保存快照", use_container_width=True):
        if has_data:
            step4_save_snapshot()
            st.rerun()

    if not has_data:
        st.info("尚无实验数据，请先在「CT反演实验」标签页执行实验流水线。")
    else:
        # 子区域1: 数据统计
        st.markdown("### 📊 数据统计")
        rows = analyzer.get_csv_rows()
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows, columns=["类别", "指标", "数值", "评价"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无统计数据")

        # 子区域2: 一致性分析
        with st.expander("### 🔍 一致性分析", expanded=False):
            patterns = analyzer.patterns or []
            if patterns:
                for p in patterns:
                    p_stripped = p.lstrip()
                    if p_stripped.startswith("✓"):
                        st.success(p)
                    elif p_stripped.startswith("✗"):
                        st.error(p)
                    elif p_stripped.startswith("~"):
                        st.warning(p)
                    else:
                        st.info(p)
            else:
                st.info("暂无一致性分析数据")

        # 子区域3: 实验结论
        with st.expander("### 📝 实验结论", expanded=False):
            conclusion = analyzer.conclusion or ""
            if conclusion:
                # 分段渲染，关键指标高亮
                for para in conclusion.split("\n"):
                    if not para.strip():
                        continue
                    # 用markdown高亮关键指标
                    highlighted = re.sub(
                        r'(RMSE|MAE)\s*[为：]?\s*([\d.]+)\s*(m/s)?',
                        r':blue[**\1=\2 \3**]', para
                    )
                    highlighted = re.sub(
                        r'损失值\s*([\d.]+)',
                        r':blue[**损失值=\1**]', highlighted
                    )
                    highlighted = re.sub(
                        r'(优秀|良好|成功)',
                        r':green[**\1**]', highlighted
                    )
                    highlighted = re.sub(
                        r'(一般|基本)',
                        r':orange[**\1**]', highlighted
                    )
                    highlighted = re.sub(
                        r'(待提高|不足|较差)',
                        r':red[**\1**]', highlighted
                    )
                    st.markdown(highlighted)
            else:
                st.info("暂无实验结论")

        # 子区域4: 可视化
        with st.expander("### 📈 可视化", expanded=False):
            viz_cols = st.columns(2)
            with viz_cols[0]:
                if st.session_state.loss_history:
                    fig = make_fig_loss(st.session_state.loss_history)
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.info("尚无训练数据")
            with viz_cols[1]:
                if st.session_state.vp is not None and st.session_state.v_true is not None:
                    fig = make_fig_error(st.session_state.vp, st.session_state.v_true)
                    st.pyplot(fig); plt.close(fig)
                else:
                    st.info("尚无反演数据")

        # 子区域5: 历史记录
        with st.expander("### 📜 历史记录", expanded=False):
            history = st.session_state.experiment_history
            if not history:
                st.info("暂无历史记录。每次反演完成后会自动保存快照。")
            else:
                # 列表
                hist_data = []
                for snap in history:
                    rating_icon = "⭐" if snap['score'] >= 80 else ("✅" if snap['score'] >= 50 else "⚠")
                    hist_data.append({
                        "#": snap['id'],
                        "时间": snap['timestamp'],
                        "评分": f"{rating_icon} {snap['score']}",
                    })
                import pandas as pd
                df_hist = pd.DataFrame(hist_data)
                sel_event = st.dataframe(df_hist, use_container_width=True, hide_index=True,
                                         selection_mode="single-row", on_select="rerun",
                                         key="history_df")

                # 选中实验详情
                selected_rows = sel_event.get("selection", {}).get("rows", []) if sel_event else []
                if selected_rows:
                    sel_idx = selected_rows[0]
                    if sel_idx < len(history):
                        snap = history[sel_idx]
                        with st.container(border=True):
                            st.markdown(f"**实验 #{snap['id']}** | {snap['timestamp']}")
                            st.markdown(f"**评分:** {snap['score']}/100 ({snap['rating']})")
                            if snap.get('acoustic_params'):
                                p = snap['acoustic_params']
                                st.markdown(f"声波: {p.get('wave_type','?')} {p.get('frequency',0)/1000:.0f} kHz")
                            inv = snap.get('inversion_stats', {})
                            if inv:
                                st.markdown(f"RMSE: {inv.get('rmse',0):.2f} m/s | MAE: {inv.get('mae',0):.2f} m/s | 相对误差: {inv.get('relative_error_pct',0):.2f}%")
                            st.markdown("**物理规律检测:**")
                            for p in snap.get('patterns', []):
                                st.caption(f"  {p}")
                            st.markdown(f"**结论:** {snap.get('conclusion','')[:200]}...")

                # 操作按钮
                hist_btns = st.columns(3)
                if hist_btns[0].button("🔄 加载选中实验", use_container_width=True):
                    # (简化: 仅恢复v_true/sino/vp/analyzer状态供查看)
                    st.info("请通过「一键执行全部」重新执行实验以获得完整数据。")
                if hist_btns[1].button("📊 比较两个实验", use_container_width=True, disabled=len(history) < 2):
                    snap_a = history[0]
                    snap_b = history[1] if len(history) > 1 else history[0]
                    cmp_cols = st.columns(2)
                    with cmp_cols[0]:
                        st.markdown(f"**实验A: #{snap_a['id']}** ({snap_a['timestamp']})")
                        st.markdown(f"评分: {snap_a['score']}/100")
                        st.caption(snap_a.get('conclusion','')[:150])
                    with cmp_cols[1]:
                        st.markdown(f"**实验B: #{snap_b['id']}** ({snap_b['timestamp']})")
                        st.markdown(f"评分: {snap_b['score']}/100")
                        st.caption(snap_b.get('conclusion','')[:150])
                if hist_btns[2].button("🗑 清空历史", use_container_width=True):
                    st.session_state.experiment_history = []
                    st.session_state.exp_counter = 0
                    st.rerun()


# ================================
# Tab 4: AI实验助手
# ================================
with tabs[3]:
    assistant = st.session_state.assistant

    # 快捷按钮
    st.markdown("### 快捷操作")
    qk_cols = st.columns(5)
    quick_actions = {
        "✅ 验证数据": "验证",
        "⭐ 评分": "评分",
        "💡 改进建议": "建议",
        "📋 完整报告": "报告",
        "🗑 清除对话": "清除",
    }
    for col, (label, action) in zip(qk_cols, quick_actions.items()):
        if col.button(label, use_container_width=True):
            if action == "清除":
                assistant.clear_chat()
                st.session_state.chat_history_msgs = []
                st.rerun()
            else:
                response = assistant.process_query(action)
                st.session_state.chat_history_msgs.append({"role": "user", "content": action})
                st.session_state.chat_history_msgs.append({"role": "assistant", "content": response})
                st.rerun()

    st.markdown("---")

    # 聊天记录
    st.markdown("### 💬 对话")
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.chat_history_msgs:
            st.info(assistant.get_welcome_message())
        else:
            for msg in st.session_state.chat_history_msgs:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])

    # 输入框
    user_input = st.chat_input("输入问题...")
    if user_input:
        response = assistant.process_query(user_input)
        st.session_state.chat_history_msgs.append({"role": "user", "content": user_input})
        st.session_state.chat_history_msgs.append({"role": "assistant", "content": response})
        st.rerun()


# ================================
# Tab 5: 虚拟实验室
# ================================
with tabs[4]:
    st.markdown("### 🔬 虚拟实验室 — 实验台操作")

    # ---- Session State 初始化 ----
    if "lab_phantom" not in st.session_state:
        st.session_state.lab_phantom = None
    if "lab_wave" not in st.session_state:
        st.session_state.lab_wave = None
    if "lab_sino" not in st.session_state:
        st.session_state.lab_sino = None
    if "lab_effects" not in st.session_state:
        st.session_state.lab_effects = None
    if "lab_summary" not in st.session_state:
        st.session_state.lab_summary = ""
    if "lab_connected" not in st.session_state:
        st.session_state.lab_connected = False

    # ---- 实验台状态栏 ----
    st.markdown("#### 实验台状态")
    status_cols = st.columns(5)
    src_on = st.session_state.lab_connected
    phantom_on = st.session_state.lab_phantom is not None
    sensor_on = st.session_state.lab_sino is not None

    status_cols[0].metric("📡 波源", "🟢 已连接" if src_on else "⚪ 待机")
    status_cols[1].markdown("<div style='text-align:center;margin-top:28px;font-size:20px;'>➡️</div>", unsafe_allow_html=True)
    status_cols[2].metric("🧫 体模", "🟢 已放置" if phantom_on else "⚪ 待放置")
    status_cols[3].markdown("<div style='text-align:center;margin-top:28px;font-size:20px;'>➡️</div>", unsafe_allow_html=True)
    status_cols[4].metric("📊 传感器", "🟢 已就绪" if sensor_on else "⚪ 未就绪")

    st.progress(
        (int(src_on) + int(phantom_on) + int(sensor_on)) / 3,
        text="实验台就绪进度"
    )

    st.markdown("---")

    # ---- 设备面板（3列卡片） ----
    st.markdown("#### 设备配置面板")
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("**📡 波源控制台**")
            wave_source = st.selectbox("选择波源",
                list(WAVE_SOURCES.keys()),
                format_func=lambda x: f"{WAVE_SOURCES[x]['icon']} {x}",
                key="lab_wave_source_v2")
            ws_info = WAVE_SOURCES[wave_source]
            st.caption(ws_info["description"])

            freq_min, freq_max = ws_info["freq_range"]
            freq_default = ws_info["freq_default"]
            lab_freq = st.slider("频率 (kHz)",
                min_value=int(freq_min/1000), max_value=int(freq_max/1000),
                value=int(freq_default/1000), step=max(1, int((freq_max-freq_min)/20000)),
                key="lab_freq_v2")
            lab_freq_hz = lab_freq * 1000
            lab_voltage = st.slider("激励电压 (V)", min_value=0.5, max_value=10.0, value=1.0, step=0.5, key="lab_voltage_v2")

            st.markdown(f"<span style='color:#6b7280;font-size:12px;'>信噪比因子: <b>{ws_info['snr_factor']*100:.0f}%</b> | 稳定性: <b>{ws_info['stability']}</b></span>", unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown("**🧫 体模舱**")
            phantom_type = st.selectbox("选择体模",
                list(PHANTOM_TYPES.keys()),
                format_func=lambda x: f"{PHANTOM_TYPES[x]['icon']} {x}",
                key="lab_phantom_type_v2")
            pt_info = PHANTOM_TYPES[phantom_type]
            st.caption(pt_info["description"])
            st.markdown(f"<span style='color:#6b7280;font-size:12px;'>异常区数量: <b>{pt_info['anomaly_count']}</b></span>", unsafe_allow_html=True)

            if phantom_type == "自定义体模":
                st.markdown("---")
                st.markdown("**自定义异常区**")
                custom_anoms = []
                for i in range(3):
                    with st.expander(f"异常区 #{i+1}", expanded=(i == 0)):
                        ea = st.checkbox(f"启用", value=(i == 0), key=f"lab_ea_enable_v2_{i}")
                        if ea:
                            cx = st.slider("中心X", -1.0, 1.0, 0.0, 0.1, key=f"lab_ea_cx_v2_{i}")
                            cy = st.slider("中心Y", -1.0, 1.0, 0.0, 0.1, key=f"lab_ea_cy_v2_{i}")
                            rad = st.slider("半径", 0.05, 0.4, 0.15, 0.05, key=f"lab_ea_r_v2_{i}")
                            spd = st.slider("声速 (m/s)", 1300, 1700, 1600, 10, key=f"lab_ea_spd_v2_{i}")
                            custom_anoms.append((cx, cy, rad, spd))
            else:
                custom_anoms = []

    with col3:
        with st.container(border=True):
            st.markdown("**📊 传感器阵列**")
            sensor_array = st.selectbox("选择阵列",
                list(SENSOR_ARRAYS.keys()),
                format_func=lambda x: f"{SENSOR_ARRAYS[x]['icon']} {x}",
                key="lab_sensor_v2")
            sa_info = SENSOR_ARRAYS[sensor_array]
            st.caption(sa_info["description"])

            st.markdown(f"<span style='color:#6b7280;font-size:12px;'>投影角度: <b>{sa_info['na']}</b> | 探测器: <b>{sa_info['ns']}</b> | 扫描: <b>{sa_info['scan_angle']}°</b></span>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**阵列参数**")
            scan_angle_override = st.slider("扫描角度 (°)", 60, 360, sa_info["scan_angle"], 10, key="lab_scan_angle_v2")

    st.markdown("---")

    # ---- 操作按钮序列 ----
    op_cols = st.columns([1, 1, 1, 3])

    with op_cols[0]:
        if st.button("🔌 连接设备", use_container_width=True, type="secondary"):
            st.session_state.lab_connected = True
            st.toast("波源设备已连接！")
            st.rerun()

    with op_cols[1]:
        can_scan = st.session_state.lab_connected
        if st.button("📡 启动扫描", use_container_width=True, type="primary", disabled=not can_scan):
            with st.spinner("正在执行扫描采集..."):
                # 生成体模
                pt_func = PHANTOM_TYPES[phantom_type]["func"]
                if pt_func == "simple":
                    st.session_state.lab_phantom = gen_phantom_simple(64)
                elif pt_func == "complex":
                    st.session_state.lab_phantom = gen_phantom_complex(64, seed=np.random.randint(0, 10000))
                elif pt_func == "layered":
                    st.session_state.lab_phantom = gen_phantom_layered(64)
                elif pt_func == "tumor":
                    st.session_state.lab_phantom = gen_phantom_tumor(64)
                elif pt_func == "custom":
                    st.session_state.lab_phantom = gen_phantom_custom(64, custom_anoms if custom_anoms else None)

                # 计算设备效果
                st.session_state.lab_effects = compute_equipment_effects(
                    wave_source, sensor_array, lab_freq_hz, lab_voltage)

                # 构建声波参数
                acoustic_params = build_acoustic_params(wave_source, lab_freq_hz, lab_voltage)

                # 计算投影
                st.session_state.lab_sino = forward_proj(
                    st.session_state.lab_phantom,
                    na=st.session_state.lab_effects["na"],
                    ns=st.session_state.lab_effects["ns"],
                    acoustic_params=acoustic_params)

                # 生成摘要
                st.session_state.lab_summary = get_equipment_summary(
                    wave_source, sensor_array, phantom_type, lab_freq_hz, lab_voltage)

                # 生成波形预览
                wtype = acoustic_params['wave_type']
                t, wave = generate_acoustic_wave(frequency=lab_freq_hz, amplitude=lab_voltage, duration=0.004, wave_type=wtype)
                st.session_state.lab_wave = (t, wave)

            st.success("扫描采集完成！体模已放置，传感器数据已就绪。")
            st.rerun()

    with op_cols[2]:
        has_lab_data = st.session_state.lab_sino is not None
        if st.button("📤 应用到流水线", use_container_width=True, disabled=not has_lab_data):
            st.session_state.v_true = st.session_state.lab_phantom
            st.session_state.sino = st.session_state.lab_sino
            st.session_state.wave_freq_val = lab_freq_hz
            st.session_state.wave_amp_val = lab_voltage
            ws_info2 = WAVE_SOURCES[wave_source]
            st.session_state.wave_type_val = ws_info2["wave_type"]
            st.session_state.analyzer.set_phantom(st.session_state.lab_phantom)
            st.session_state.analyzer.set_sino(st.session_state.lab_sino, build_acoustic_params(wave_source, lab_freq_hz, lab_voltage))
            for s in st.session_state.pipeline_steps:
                s["state"] = "pending"
            st.session_state.pipeline_steps[0]["state"] = "completed"
            st.session_state.pipeline_steps[1]["state"] = "completed"
            st.session_state.vp = None
            st.session_state.loss_history = []
            st.session_state.status_message = "已加载虚拟实验室数据，请执行AI反演（第3步）"
            st.success("数据已加载！切换到「CT反演实验」继续。")
            st.rerun()

    # ---- 数据预览区 ----
    if st.session_state.lab_sino is not None:
        st.markdown("---")
        st.markdown("### 📊 扫描结果预览")

        preview_cols = st.columns(3)

        with preview_cols[0]:
            if st.session_state.lab_phantom is not None:
                fig = make_fig_phantom(st.session_state.lab_phantom)
                fig.axes[0].set_title(f"声速场: {phantom_type}", fontsize=10)
                st.pyplot(fig); plt.close(fig)

        with preview_cols[1]:
            if st.session_state.lab_sino is not None:
                fig = make_fig_sino(st.session_state.lab_sino)
                fig.axes[0].set_title(f"投影: {sensor_array}", fontsize=10)
                st.pyplot(fig); plt.close(fig)

        with preview_cols[2]:
            if st.session_state.lab_wave is not None:
                t, wave = st.session_state.lab_wave
                fig = make_fig_waveform(t, wave, f"波形: {wave_source} {lab_freq}kHz")
                st.pyplot(fig); plt.close(fig)

        # 统计指标
        stat_cols = st.columns(4)
        eff = st.session_state.lab_effects
        if eff:
            stat_cols[0].metric("信噪比", f"{eff['snr_db']:.1f} dB")
            stat_cols[1].metric("分辨率", f"{eff['resolution_mm']:.2f} mm")
            stat_cols[2].metric("数据量", f"{eff['na']}×{eff['ns']}")
            stat_cols[3].metric("衰减系数", f"{eff['attenuation_factor']:.4f}")

        with st.expander("📋 设备配置详细报告"):
            st.code(st.session_state.lab_summary, language=None)
    else:
        if not st.session_state.lab_connected:
            st.info("👆 请先配置设备参数，点击「🔌 连接设备」启动实验台。")
        else:
            st.info("👆 设备已连接，点击「📡 启动扫描」开始采集数据。")

# ====================================================================
# 底部
# ====================================================================
st.markdown("---")
st.caption("🔬 AI驱动声波CT反演实验系统 | 基于Streamlit | 核心计算使用PyTorch + NumPy")
