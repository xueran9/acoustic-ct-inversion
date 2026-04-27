"""实验报告导出功能：TXT / CSV"""
import csv
import numpy as np


def export_txt_report(analyzer, assistant, filepath):
    """导出完整实验报告为TXT文件"""
    assistant.refresh()
    report = analyzer.get_report_dict()
    stats = report['stats']
    lines = []
    lines.append("=" * 50)
    lines.append("  声波CT反演实验报告")
    lines.append("=" * 50)
    lines.append("")

    # 实验参数
    lines.append("【实验参数】")
    if analyzer.acoustic_params:
        p = analyzer.acoustic_params
        lines.append(f"  波形类型: {p.get('wave_type', '未知')}")
        lines.append(f"  频率: {p.get('frequency', 0)/1000:.1f} kHz")
        lines.append(f"  振幅: {p.get('amplitude', 1.0):.2f}")
    lines.append("")

    # 声速场统计
    if analyzer.has_phantom():
        lines.append("【声速场统计】")
        lines.append(f"  最小值: {stats.get('v_min',0):.2f} m/s")
        lines.append(f"  最大值: {stats.get('v_max',0):.2f} m/s")
        lines.append(f"  均值: {stats.get('v_mean',0):.2f} m/s")
        lines.append(f"  标准差: {stats.get('v_std',0):.2f} m/s")
        lines.append(f"  梯度均值: {stats.get('grad_mean',0):.2f}")
        lines.append(f"  异常区域占比: {stats.get('anomaly_pct',0):.1f}%")
        lines.append("")

    # 投影数据统计
    if analyzer.has_sino():
        lines.append("【投影数据统计】")
        lines.append(f"  动态范围: {stats.get('dynamic_range',0):.6f} s")
        lines.append(f"  信噪比(SNR): {stats.get('snr_estimate',0):.2f}")
        lines.append(f"  角度一致性: {stats.get('mean_cross_corr',0):.4f}")
        lines.append("")

    # 反演统计
    if analyzer.has_inversion():
        lines.append("【反演结果统计】")
        lines.append(f"  RMSE: {stats.get('rmse',0):.2f} m/s")
        lines.append(f"  MAE: {stats.get('mae',0):.2f} m/s")
        lines.append(f"  相对误差: {stats.get('relative_error_pct',0):.2f}%")
        lines.append(f"  损失终值: {stats.get('loss_final',0):.6f}")
        lines.append(f"  损失下降比: {stats.get('loss_decay_ratio',0):.4f}")
        lines.append("")

    # 物理规律检测
    lines.append("【物理规律检测】")
    for p in report['patterns']:
        lines.append(f"  {p}")
    lines.append("")

    # AI评估
    lines.append("【AI评估】")
    lines.append(f"  综合评分: {assistant.experiment_score or 0}/100")
    lines.append(f"  质量等级: {assistant.quality_rating or '无数据'}")
    if assistant.issues:
        for issue in assistant.issues:
            icon = "❌" if issue['severity'] == 'error' else "⚠"
            lines.append(f"  {icon} {issue['message']}")
    lines.append("")

    # 改进建议
    suggestions = assistant.get_suggestions()
    if suggestions:
        lines.append("【改进建议】")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"  {i}. {s}")
        lines.append("")

    # 实验结论
    lines.append("【实验结论】")
    lines.append(f"  {report['conclusion']}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("  报告结束")
    lines.append("=" * 50)

    content = "\n".join(lines)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath


def export_csv_report(analyzer, filepath):
    """导出统计数据为CSV文件"""
    rows = analyzer.get_csv_rows()
    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["类别", "指标", "数值", "评价"])
        for row in rows:
            writer.writerow(row)
    return filepath
