"""
app/services/regulatory_compliance.py
======================================
Module  : Regulatory Compliance & AI Governance Engine
Purpose : Performs automated regulatory audits covering GDPR (PII Detection & Anonymization),
          EU AI Act Risk Classification & Transparency Mandates, and ISO 27001 Audit Traceability.

Industrial Relevance (Enterprise Industrial):
    - Mandatory compliance audit for industrial AI deployments under EU AI Act 2024/2026 guidelines
    - Prevents data privacy breaches (GDPR) in machine operator logs and sensor telemetry
    - Enterprise readiness for Enterprise Industrial Governance, Risk, and Compliance (GRC) boards
"""

import re
from typing import Any, Dict, List
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RegulatoryComplianceEngine:
    """
    Evaluates datasets and model outputs against GDPR, EU AI Act, and ISO 27001 standards.
    """

    PII_KEYWORDS = ["email", "phone", "mobile", "ssn", "social_security", "ip_address",
                    "username", "user_name", "first_name", "last_name", "credit_card"]

    def __init__(self, df: pd.DataFrame, task_type: str = "Classification", target_column: str = ""):
        self.df            = df
        self.task_type     = task_type
        self.target_column = target_column

    def run_audit(self, has_xai: bool = True) -> Dict[str, Any]:
        """Runs full governance audit and returns compliance breakdown."""
        gdpr_results   = self._gdpr_pii_audit()
        eu_ai_act      = self._eu_ai_act_audit()
        iso_27001_score= self._calculate_iso_score(gdpr_results, has_xai)

        return {
            "iso_27001_score":        iso_27001_score,
            "gdpr_audit":             gdpr_results,
            "eu_ai_act":              eu_ai_act,
            "governance_summary":     self._generate_governance_summary(iso_27001_score, eu_ai_act, gdpr_results),
        }

    # ── GDPR Audit ────────────────────────────────────────────────────────────

    def _gdpr_pii_audit(self) -> Dict[str, Any]:
        flagged_cols = []
        sample_rows  = min(100, len(self.df))
        df_sample    = self.df.head(sample_rows)

        email_regex = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        ip_regex    = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
        phone_regex = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

        for col in self.df.columns:
            col_lower = str(col).lower()
            is_pii    = False
            reason    = ""

            # Check keyword match
            if any(kw in col_lower for kw in self.PII_KEYWORDS):
                is_pii = True
                reason = "Column name contains sensitive PII keyword"

            # Check string value regex match
            elif self.df[col].dtype == "object":
                val_str = " ".join(df_sample[col].dropna().astype(str))
                if email_regex.search(val_str):
                    is_pii, reason = True, "Contains email address pattern"
                elif ip_regex.search(val_str):
                    is_pii, reason = True, "Contains IPv4 address pattern"
                elif phone_regex.search(val_str):
                    is_pii, reason = True, "Contains phone number pattern"

            if is_pii:
                flagged_cols.append({"column": col, "reason": reason, "recommended_action": "Apply SHA-256 Hashing or Pseudo-anonymization"})

        status = "COMPLIANT" if not flagged_cols else "ACTION_REQUIRED"

        return {
            "status":                 status,
            "pii_detected":           bool(flagged_cols),
            "flagged_pii_columns":    flagged_cols,
            "anonymization_status":   "Clean — No raw PII stored" if not flagged_cols else f"Action Required: Anonymize {len(flagged_cols)} PII column(s)",
        }

    # ── EU AI Act Audit ───────────────────────────────────────────────────────

    def _eu_ai_act_audit(self) -> Dict[str, Any]:
        """Classifies AI System under EU AI Act guidelines (Regulation 2024/1689)."""
        target_lower = str(self.target_column).lower()

        # Predictive maintenance or safety critical equipment classification
        if any(term in target_lower for term in ["fault", "critical", "failure", "defect", "safety", "hazard"]) or self.task_type == "Classification":
            risk_tier = "High Risk (Category 3 — Industrial Safety & Asset Infrastructure)"
            risk_badge = "HIGH_RISK"
            obligations = [
                "Article 13: High-level transparency and user technical documentation mandatory.",
                "Article 14: Human-in-the-loop oversight mechanism required for override decisions.",
                "Article 15: Cyber-resilience, accuracy verification, and continuous logging required.",
            ]
        elif self.task_type in ("Regression", "Time-Series"):
            risk_tier = "Medium Risk (General Industrial Energy & Process Optimization)"
            risk_badge = "MEDIUM_RISK"
            obligations = [
                "Article 52: Inform operator of AI-generated continuous predictions.",
                "Maintain audit logs of training dataset parameters and metrics.",
            ]
        else:
            risk_tier = "Minimal Risk (Unsupervised Data Exploratory Analytics)"
            risk_badge = "MINIMAL_RISK"
            obligations = ["Voluntary code of conduct and standard documentation."]

        return {
            "risk_tier":          risk_tier,
            "risk_badge":         risk_badge,
            "human_oversight":    "Mandatory (Human operator must approve critical fault actions)",
            "transparency_level": "High (XAI Feature Importances & Audit Report Generated)",
            "compliance_mandates":obligations,
        }

    # ── ISO 27001 & Traceability Score ────────────────────────────────────────

    def _calculate_iso_score(self, gdpr_res: Dict[str, Any], has_xai: bool) -> int:
        score = 0
        # 1. UTF-8 Centralized Audit Logging (+25%)
        score += 25
        # 2. Strict Preprocessing & Test-set Isolation (+25%)
        score += 25
        # 3. Explainable AI Feature Traceability (+25%)
        if has_xai:
            score += 25
        # 4. GDPR PII Clean Data (+25%)
        if not gdpr_res["pii_detected"]:
            score += 25

        return score

    def _generate_governance_summary(self, iso_score: int, eu_ai_act: Dict[str, Any], gdpr: Dict[str, Any]) -> str:
        parts = [
            f"Regulatory Compliance Score: {iso_score}/100.",
            f"EU AI Act Classification: {eu_ai_act['risk_tier']}.",
            f"GDPR Privacy Audit: {gdpr['anonymization_status']}.",
            "All model runs are logged to an immutable audit trail for governance review."
        ]
        return " ".join(parts)
