"""
app/services/expertise_adaptation.py
======================================
Module  : User Expertise Adaptation Module
Purpose : Dynamically rewrites AI insights at three comprehension levels:
          Beginner → Intermediate → Expert

Design:
    - Each insight category has a dedicated transformer per level
    - Context-aware: different keys produce different simplifications
    - Easily extensible for new insight categories

Industrial Relevance (Enterprise Industrial):
    - Floor technicians need simple red/green indicators
    - Process engineers need moderate technical detail
    - Data scientists need full metric visibility
"""

from typing import Dict, Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Level Constants ───────────────────────────────────────────────────────────
LEVEL_BEGINNER     = "beginner"
LEVEL_INTERMEDIATE = "intermediate"
LEVEL_EXPERT       = "expert"


class ExpertiseAdapter:
    """
    Rewrites base insights to match the user's expertise level.

    Usage:
        adapted = ExpertiseAdapter.adapt_insights(base_insights, level="beginner")
    """

    @staticmethod
    def adapt_insights(base_insights: Dict[str, str], level: str) -> Dict[str, str]:
        """
        Adapts all insight sections to the specified expertise level.

        Args:
            base_insights : {key: text} dict from InsightGenerator
            level         : "beginner" | "intermediate" | "expert"

        Returns:
            Adapted {key: text} dict
        """
        if level not in [LEVEL_BEGINNER, LEVEL_INTERMEDIATE, LEVEL_EXPERT]:
            logger.warning(f"Unknown level '{level}' — defaulting to intermediate.")
            level = LEVEL_INTERMEDIATE

        adapted = {}
        for key, text in base_insights.items():
            if level == LEVEL_BEGINNER:
                adapted[key] = ExpertiseAdapter._beginner(key, text)
            elif level == LEVEL_EXPERT:
                adapted[key] = ExpertiseAdapter._expert(key, text)
            else:
                adapted[key] = text  # Intermediate uses the natural-language base text

        logger.info(f"Insights adapted for '{level}' user.")
        return adapted

    # ── Beginner Transformers ─────────────────────────────────────────────────

    @staticmethod
    def _beginner(context: str, text: str) -> str:
        """
        Strips jargon. Uses analogies and plain everyday language.
        Target: plant floor operators, students, non-technical stakeholders.
        """
        templates = {
            "data_profile": (
                "We uploaded your spreadsheet and checked it carefully. "
                "If there were any blank cells, we filled them in automatically "
                "so the AI could understand everything. Your data is now ready to learn from! 📊"
            ),
            "task": (
                "We figured out what kind of question you are asking the AI. "
                + _beginner_task_hint(text)
            ),
            "model": (
                "We tested several AI 'brains' and picked the one that made "
                "the fewest mistakes on your data. Think of it like picking the best student "
                "for an exam — the one who got the most right answers wins! 🏆"
            ),
            "features": (
                "The AI noticed that a few columns in your spreadsheet are the most "
                "important clues for making a good guess. It focuses on those the most, "
                "just like a detective looking for the strongest evidence. 🔍"
            ),
            "preprocessing": (
                "Before training, we cleaned up the data — removed duplicates, "
                "filled in blanks, and converted text labels to numbers so the AI can read them."
            ),
        }
        return templates.get(context, text)

    # ── Expert Transformers ────────────────────────────────────────────────────

    @staticmethod
    def _expert(context: str, text: str) -> str:
        """
        Appends technical depth: statistical detail, architecture notes, engineering caveats.
        Target: data scientists, ML engineers, senior researchers.
        """
        appendages = {
            "data_profile": (
                " | Pipeline: median imputation for continuous features (resistant to skew), "
                "mode imputation for categoricals. IQR-based outlier flagging applied. "
                "One-Hot Encoding via pd.get_dummies (drop_first=True to avoid multicollinearity)."
            ),
            "task": (
                " | Stratified 80/20 train-test split applied (stratify=y for classification). "
                "No data leakage — preprocessing fit exclusively on training fold."
            ),
            "model": (
                " | Model selection criterion: weighted F1 (classification) or RMSE (regression). "
                "All models trained with default hyperparameters; recommend GridSearchCV / "
                "Optuna for production hyperparameter optimisation."
            ),
            "features": (
                " | Importances sourced from estimator's native feature_importances_ "
                "(tree-based) or mean(|coef_|) (linear). "
                "For model-agnostic explanations, integrate SHAP TreeExplainer / KernelExplainer."
            ),
            "preprocessing": (
                " | StandardScaler fitted on X_train only; transform applied to both train/test "
                "to prevent test-set leakage. Boolean dummy columns cast to int8 for memory efficiency."
            ),
        }
        suffix = appendages.get(context, "")
        return text + suffix


# ── Helper ────────────────────────────────────────────────────────────────────

def _beginner_task_hint(text: str) -> str:
    """Detects task type from the insight text and returns a beginner-friendly hint."""
    if "Regression" in text:
        return (
            "It looks like you want to predict a number, like a temperature, price, or speed. "
            "We set up the AI to give you number predictions! 🔢"
        )
    if "Classification" in text:
        return (
            "It looks like you want to sort things into groups, like 'Broken' or 'Working'. "
            "We set up the AI to make these yes/no or category decisions! ✅❌"
        )
    if "Clustering" in text:
        return (
            "You didn't tell us what to predict, so the AI will look for hidden patterns "
            "and automatically group similar items together! 🔵🟡🔴"
        )
    if "Time-Series" in text:
        return (
            "It looks like you have data over time (like daily readings). "
            "The AI will learn the pattern and predict what comes next! 📅"
        )
    return text
