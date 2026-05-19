"""
app/services/report_generator.py
==================================
Module  : Report Generation System
Purpose : Generates a structured PDF report summarising the full ADSLM pipeline run.

Output:
    A timestamped PDF saved to reports/ directory containing:
    - Dataset profile
    - Task type and reasoning
    - Preprocessing summary
    - Model recommendations
    - Performance metrics table
    - Top feature importances
    - AI insights and recommendations

Industrial Relevance (ABB):
    - Provides a traceable audit trail for AI-assisted decisions
    - Submission-ready documentation for engineering review boards
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import REPORTS_DIR
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates a plain-text or (optionally) PDF report of the ADSLM pipeline output.

    Usage:
        gen = ReportGenerator(pipeline_results)
        path = gen.generate()
    """

    def __init__(self, pipeline_results: Dict[str, Any]):
        self.results   = pipeline_results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> str:
        """
        Generates both a JSON and a formatted text report.

        Returns:
            Path to the generated text report.
        """
        report_text = self._build_text_report()
        report_json = self.results

        txt_path  = REPORTS_DIR / f"adslm_report_{self.timestamp}.txt"
        json_path = REPORTS_DIR / f"adslm_report_{self.timestamp}.json"

        txt_path.write_text(report_text, encoding="utf-8")
        json_path.write_text(
            json.dumps(report_json, indent=2, default=str), encoding="utf-8"
        )

        # Attempt PDF generation (requires reportlab)
        pdf_path = self._try_generate_pdf(report_text)

        logger.info(f"Report saved → {txt_path}")
        return str(pdf_path if pdf_path else txt_path)

    def get_report_text(self) -> str:
        """Returns the formatted text report as a string (for Streamlit display)."""
        return self._build_text_report()

    # ── Private Builders ──────────────────────────────────────────────────────

    def _build_text_report(self) -> str:
        meta     = self.results.get("metadata", {})
        recs     = self.results.get("recommendations", [])
        metrics  = self.results.get("metrics", {})
        insights = self.results.get("insights", {})
        prep     = self.results.get("preprocessing_summary", {})
        actions  = self.results.get("actionable_recommendations", [])
        feat_imp = self.results.get("feature_importances", {})

        lines = [
            "=" * 70,
            "  ADSLM — ADAPTIVE DATA SCIENCE LANGUAGE MODEL",
            "  ABB Innovation Evaluation Report",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "── PIPELINE OVERVIEW ─────────────────────────────────────────────",
            f"  Task Type    : {meta.get('task_type', 'N/A')}",
            f"  Target Column: {meta.get('target_column', 'None (Clustering)')}",
            f"  Best Model   : {meta.get('best_model', 'N/A')}",
            f"  Best Score   : {meta.get('best_score', 'N/A')}",
            "",
            "── DATASET INSIGHT ───────────────────────────────────────────────",
            f"  {insights.get('data_profile', '')}",
            "",
            "── TASK DETECTION ────────────────────────────────────────────────",
            f"  {insights.get('task', '')}",
            "",
            "── PREPROCESSING SUMMARY ─────────────────────────────────────────",
            f"  {insights.get('preprocessing', 'Standard pipeline applied.')}",
            "",
            "── MODEL RECOMMENDATIONS ─────────────────────────────────────────",
        ]
        for rec in recs:
            lines.append(f"  [{rec.get('priority', '?')}] {rec['model']}")
            lines.append(f"      → {rec['reason']}")
        lines.append("")

        lines += [
            "── PERFORMANCE METRICS ───────────────────────────────────────────",
        ]
        for metric, value in metrics.items():
            if metric == "Confusion Matrix":
                lines.append(f"  Confusion Matrix:")
                for row in value:
                    lines.append(f"      {row}")
            else:
                lines.append(f"  {metric:<25}: {value}")
        lines.append("")

        if feat_imp and "_note" not in feat_imp and "error" not in feat_imp:
            lines += ["── TOP FEATURE IMPORTANCES ───────────────────────────────────────"]
            for i, (feat, score) in enumerate(list(feat_imp.items())[:10], 1):
                bar = "█" * int(score * 40)
                lines.append(f"  {i:>2}. {feat:<30} {score:.4f}  {bar}")
            lines.append("")

        lines += [
            "── AI MODEL INSIGHT ──────────────────────────────────────────────",
            f"  {insights.get('model', '')}",
            "",
            "── FEATURE INSIGHT ───────────────────────────────────────────────",
            f"  {insights.get('features', '')}",
            "",
            "── ACTIONABLE RECOMMENDATIONS ────────────────────────────────────",
        ]
        for i, rec in enumerate(actions, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")
        lines += [
            "=" * 70,
            "  End of ADSLM Report | ABB Evaluation Submission",
            "=" * 70,
        ]

        return "\n".join(lines)

    def _try_generate_pdf(self, report_text: str) -> Path | None:
        """Attempts to generate a PDF using reportlab if available."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib import colors

            pdf_path = REPORTS_DIR / f"adslm_report_{self.timestamp}.pdf"

            doc    = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                       leftMargin=2*cm, rightMargin=2*cm,
                                       topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                "TitleStyle",
                parent    = styles["Title"],
                textColor = colors.HexColor("#CC0000"),
                fontSize  = 18,
                spaceAfter= 12,
            )
            body_style = ParagraphStyle(
                "BodyStyle",
                parent   = styles["Normal"],
                fontSize = 10,
                leading  = 14,
                fontName = "Courier",
            )

            story = [
                Paragraph("ADSLM — Adaptive Data Science Language Model", title_style),
                Paragraph("ABB Innovation Evaluation Report", styles["Heading2"]),
                Spacer(1, 0.5*cm),
            ]

            for line in report_text.split("\n"):
                safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe_line or "&nbsp;", body_style))

            doc.build(story)
            logger.info(f"PDF report generated → {pdf_path}")
            return pdf_path

        except ImportError:
            logger.warning("reportlab not installed — PDF generation skipped. Only TXT/JSON reports created.")
            return None
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None
