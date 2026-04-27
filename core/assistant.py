"""AI实验助手模块：规则引擎驱动的数据合理性验证与评分"""
from .analyzer import DataAnalyzer


class AIAssistant:
    """基于规则引擎的AI实验助手：验证数据合理性、评分、提供改进建议"""
    def __init__(self, analyzer: DataAnalyzer):
        self.analyzer = analyzer
        self.chat_history = []
        self.experiment_score = None
        self.quality_rating = None
        self.issues = []

    def refresh(self):
        self.issues = self.validate_reasonability()
        self.compute_score()

    def validate_reasonability(self):
        issues = []
        analyzer = self.analyzer

        if not analyzer.has_phantom():
            issues.append({"type": "data_missing", "severity": "error", "field": "phantom",
                           "message": "尚未生成模拟声场，请先执行步骤1"})
        else:
            stats = analyzer.phantom_stats or analyzer.compute_phantom_stats()
            if not stats.get('physically_valid', True):
                issues.append({"type": "physical_range", "severity": "error", "field": "phantom",
                               "message": f"声速值{stats['v_min']:.1f}-{stats['v_max']:.1f} m/s超出生物组织典型范围(1000-2000 m/s)"})
            if stats.get('anomaly_pct', 0) > 30:
                issues.append({"type": "phantom_quality", "severity": "warning", "field": "phantom",
                               "message": f"异常区域占比{stats['anomaly_pct']:.1f}%，体模可能过于复杂"})

        if not analyzer.has_sino():
            issues.append({"type": "data_missing", "severity": "warning", "field": "sino",
                           "message": "尚未生成声波投影，将仅分析部分数据"})
        else:
            stats = analyzer.sino_stats or analyzer.compute_sino_stats()
            if not stats.get('is_positive', True):
                issues.append({"type": "data_quality", "severity": "error", "field": "sino",
                               "message": "投影数据包含非正值，物理上不合理的走时"})
            snr = stats.get('snr_estimate', 0)
            if snr < 3:
                issues.append({"type": "data_quality", "severity": "warning", "field": "sino",
                               "message": f"投影数据信噪比较低({snr:.1f})，可能需要调整声波参数"})

        if not analyzer.has_inversion():
            issues.append({"type": "data_missing", "severity": "warning", "field": "inversion",
                           "message": "尚未执行AI反演，将仅分析部分数据"})
        else:
            stats = analyzer.inversion_stats or analyzer.compute_inversion_stats()
            rmse = stats.get('rmse', 999)
            if rmse > 50:
                issues.append({"type": "inversion_quality", "severity": "error", "field": "inversion",
                               "message": f"反演RMSE较大({rmse:.1f} m/s)，重建精度不足"})
            elif rmse > 20:
                issues.append({"type": "inversion_quality", "severity": "warning", "field": "inversion",
                               "message": f"反演精度一般(RMSE={rmse:.1f} m/s)，可优化"})
            if stats.get('loss_decay_ratio', 1) > 0.5:
                issues.append({"type": "convergence", "severity": "error", "field": "inversion",
                               "message": f"损失函数未有效收敛(下降比{stats['loss_decay_ratio']:.2f})，建议检查网络或数据"})
            elif stats.get('loss_decay_ratio', 1) > 0.3:
                issues.append({"type": "convergence", "severity": "warning", "field": "inversion",
                               "message": f"损失函数下降不足(下降比{stats['loss_decay_ratio']:.2f})，可能需要更多训练轮数"})

        if analyzer.acoustic_params:
            freq = analyzer.acoustic_params.get('frequency', 50000)
            if freq > 150000:
                issues.append({"type": "parameter", "severity": "warning", "field": "wave",
                               "message": f"高频({freq:.0f} Hz)可能导致严重衰减，影响投影质量"})
            elif freq < 5000:
                issues.append({"type": "parameter", "severity": "warning", "field": "wave",
                               "message": f"低频({freq:.0f} Hz)可能导致分辨率不足"})

        self.issues = issues
        return issues

    def compute_score(self):
        errors = sum(1 for i in self.issues if i['severity'] == 'error')
        warnings = sum(1 for i in self.issues if i['severity'] == 'warning')
        score = 100 - errors * 25 - warnings * 10
        score = max(0, min(100, score))
        self.experiment_score = score
        if score >= 80:
            self.quality_rating = "优秀"
        elif score >= 50:
            self.quality_rating = "良好"
        else:
            self.quality_rating = "较差"
        return score, self.quality_rating

    def get_suggestions(self):
        suggestions = []
        seen_types = set()
        for issue in self.issues:
            if issue['type'] not in seen_types:
                seen_types.add(issue['type'])
                if issue['type'] == 'data_missing':
                    suggestions.append("请完成所有实验步骤：生成声场 → 投影 → 反演，以获得完整分析。")
                elif issue['type'] == 'physical_range':
                    suggestions.append("检查体模参数，确保声速值在1000-2000 m/s生物组织典型范围内。")
                elif issue['type'] == 'data_quality':
                    suggestions.append("尝试调整声波频率或振幅，或更换波形类型以改善投影数据质量。")
                elif issue['type'] == 'inversion_quality':
                    suggestions.append("考虑增加训练轮数(当前200轮)、减小学习率或增加网络容量(更多隐藏层神经元)。")
                elif issue['type'] == 'convergence':
                    suggestions.append("损失未充分收敛，建议增加训练轮数(如从200增至500)或降低学习率。")
                elif issue['type'] == 'phantom_quality':
                    suggestions.append("考虑简化体模模型，减少异常区域数量或增大平滑尺度。")
                elif issue['type'] == 'parameter':
                    suggestions.append("建议将声波频率调整至20kHz-100kHz范围以获得更优成像效果。")
        if not suggestions and self.issues:
            suggestions.append("当前无明显问题，可以尝试更多实验参数组合以探索最优配置。")
        if not self.issues:
            suggestions.append("尚无实验数据，请先进行CT反演实验。")
        return suggestions

    def get_validation_report(self):
        self.refresh()
        if not self.issues:
            return "✅ 数据合理性验证\n未发现明显问题。所有数据在物理合理范围内，实验设置正确。"
        lines = ["🔍 数据合理性验证结果："]
        errors = [i for i in self.issues if i['severity'] == 'error']
        warnings = [i for i in self.issues if i['severity'] == 'warning']
        if errors:
            lines.append(f"\n【严重问题】({len(errors)}项)")
            for e in errors:
                lines.append(f"  ❌ {e['message']}")
        if warnings:
            lines.append(f"\n【警告】({len(warnings)}项)")
            for w in warnings:
                lines.append(f"  ⚠ {w['message']}")
        return "\n".join(lines)

    def get_quality_report(self):
        self.refresh()
        score = self.experiment_score or 0
        rating = self.quality_rating or "无数据"
        if not self.analyzer.has_phantom() and not self.analyzer.has_sino():
            return "📊 尚无实验数据，请先完成CT反演实验步骤后再进行评分。"
        lines = [f"📊 实验质量评分: {score}/100 — 「{rating}」"]
        errors = sum(1 for i in self.issues if i['severity'] == 'error')
        warnings = sum(1 for i in self.issues if i['severity'] == 'warning')
        lines.append(f"评分依据: {errors}个严重问题, {warnings}个警告")
        if score >= 80:
            lines.append("总体评价: 实验设置合理，数据质量良好。")
        elif score >= 50:
            lines.append("总体评价: 实验基本可用，但存在优化空间。")
        else:
            lines.append("总体评价: 实验存在较多问题，建议调整后重新实验。")
        return "\n".join(lines)

    def get_suggestions_text(self):
        suggestions = self.get_suggestions()
        if not suggestions:
            return "💡 当前无待改进项。"
        lines = ["💡 改进建议："]
        for i, s in enumerate(suggestions, 1):
            lines.append(f"  {i}. {s}")
        return "\n".join(lines)

    def get_full_report(self):
        self.refresh()
        lines = ["📋 完整实验报告"]
        lines.append("=" * 40)
        lines.append("")
        lines.append("【数据概况】")
        if self.analyzer.has_phantom():
            s = self.analyzer.phantom_stats or self.analyzer.compute_phantom_stats()
            lines.append(f"  声速场: {s.get('v_min',0):.0f}-{s.get('v_max',0):.0f} m/s, "
                        f"均值{s.get('v_mean',0):.0f} m/s, 标准差{s.get('v_std',0):.1f}")
        if self.analyzer.has_sino():
            s = self.analyzer.sino_stats or self.analyzer.compute_sino_stats()
            lines.append(f"  投影数据: 范围{s.get('s_min',0):.4f}-{s.get('s_max',0):.4f} s, "
                        f"SNR={s.get('snr_estimate',0):.1f}")
        if self.analyzer.has_inversion():
            s = self.analyzer.inversion_stats or self.analyzer.compute_inversion_stats()
            lines.append(f"  反演结果: RMSE={s.get('rmse',0):.2f} m/s, "
                        f"MAE={s.get('mae',0):.2f} m/s, 损失终值={s.get('loss_final',0):.6f}")
        lines.append("")
        score = self.experiment_score or 0
        rating = self.quality_rating or "无数据"
        lines.append(f"【质量评分】{score}/100 — 「{rating}」")
        lines.append("")
        if self.issues:
            lines.append("【发现的问题】")
            for issue in self.issues:
                icon = "❌" if issue['severity'] == 'error' else "⚠"
                lines.append(f"  {icon} {issue['message']}")
            lines.append("")
        suggestions = self.get_suggestions()
        if suggestions:
            lines.append("【改进建议】")
            for i, s in enumerate(suggestions, 1):
                lines.append(f"  {i}. {s}")
        return "\n".join(lines)

    def process_query(self, user_input):
        text = user_input.strip()
        if not text:
            return "请输入您的问题。"

        intent = None
        greetings = ["你好", "您好", "hi", "hello", "help", "帮助", "使用"]
        validate_kw = ["验证", "合理", "检查", "valid", "check"]
        quality_kw = ["质量", "评分", "评分", "score", "评价", "怎么样", "如何"]
        suggest_kw = ["改进", "建议", "优化", "suggest", "advice", "改善", "提高"]
        report_kw = ["报告", "完整", "report", "summary", "概况", "总览"]

        if any(k in text for k in validate_kw):
            intent = "VALIDATE"
        elif any(k in text for k in quality_kw):
            intent = "QUALITY"
        elif any(k in text for k in suggest_kw):
            intent = "SUGGEST"
        elif any(k in text for k in report_kw):
            intent = "REPORT"
        elif any(k in text for k in greetings):
            intent = "GREETING"

        if intent is None:
            response = ("\U0001f916 我是AI实验助手，您可以询问我以下内容：\n"
                        "  • 验证数据合理性 — 输入「验证」\n"
                        "  • 实验质量评分 — 输入「评分」\n"
                        "  • 改进建议 — 输入「建议」\n"
                        "  • 完整实验报告 — 输入「报告」\n"
                        "也可以直接点击下方的快捷按钮。")
        elif intent == "GREETING":
            response = ("你好！我是AI实验助手，可以帮你：\n"
                        "1. ✅ 验证实验数据的合理性\n"
                        "2. ⭐ 评估实验质量并评分\n"
                        "3. 💡 提供改进建议\n"
                        "4. 📋 生成完整实验报告\n"
                        "请告诉我你需要什么帮助，或点击下方快捷按钮。")
        elif intent == "VALIDATE":
            response = self.get_validation_report()
        elif intent == "QUALITY":
            response = self.get_quality_report()
        elif intent == "SUGGEST":
            response = self.get_suggestions_text()
        elif intent == "REPORT":
            response = self.get_full_report()
        else:
            response = "请确认您的问题。"

        self.chat_history.append(("用户", text))
        self.chat_history.append(("助手", response))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        return response

    def get_welcome_message(self):
        return ("🤖 欢迎使用AI实验助手！我可以：\n"
                "  1. 验证实验数据的合理性\n"
                "  2. 评估实验质量并评分\n"
                "  3. 提供改进建议\n"
                "  4. 生成完整实验报告\n"
                "请在下方输入您的问题，或点击快捷按钮。")

    def get_chat_display_text(self):
        if not self.chat_history:
            return self.get_welcome_message()
        lines = []
        for speaker, text in self.chat_history:
            if speaker == "系统":
                lines.append(f"[系统] {text}")
            elif speaker == "用户":
                lines.append(f"[用户] {text}")
            elif speaker == "助手":
                lines.append(f"[助手] {text}")
            lines.append("-" * 50)
        return "\n".join(lines)

    def clear_chat(self):
        self.chat_history = []
        self.issues = []
        self.experiment_score = None
        self.quality_rating = None
