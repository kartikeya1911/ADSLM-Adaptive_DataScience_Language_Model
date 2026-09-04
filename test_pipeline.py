"""End-to-end smoke test for ADSLM pipeline"""
import io
import pandas as pd
from app.services.dataset_analyzer import DatasetAnalyzer
from app.services.task_detection import TaskDetector
from app.services.preprocessing import PreprocessingEngine
from app.services.model_recommendation import ModelRecommendationEngine
from app.services.training import TrainingEngine
from app.services.explainability import ExplainabilityEngine
from app.services.insight_generator import InsightGenerator
from app.services.expertise_adaptation import ExpertiseAdapter

df = pd.read_csv("datasets/predictive_maintenance.csv")
buf = io.StringIO(df.to_csv(index=False))

# 1. Analyze
analyzer = DatasetAnalyzer(buf)
stats = analyzer.analyze()
print("Rows:", stats["basic_info"]["row_count"], "Cols:", stats["basic_info"]["column_count"])

# 2. Detect
detector = TaskDetector(stats, "Fault")
task = detector.detect_task()
print("Task:", task)

# 3. Preprocess
preprocessor = PreprocessingEngine(df, "Fault", task)
X_train, X_test, y_train, y_test = preprocessor.preprocess()
print("X_train shape:", X_train.shape)

# 4. Recommend
recs = ModelRecommendationEngine(task, stats).recommend()
print("Models:", [r["model"] for r in recs])

# 5. Train
trainer = TrainingEngine(task, X_train, y_train, X_test, y_test, [r["model"] for r in recs])
results = trainer.run()
best = results["best_model"]
score = results["best_score_metric"]
print(f"Best model: {best} | Score: {score:.4f}")

# 6. XAI
xai = ExplainabilityEngine(trainer.model_registry[best], X_train.columns.tolist())
top = xai.get_top_features(3)
print("Top features:", [t["feature"] for t in top])

# 7. Insights + Adaptation
insights = {
    "data_profile": InsightGenerator.generate_data_profile_insights(stats),
    "task": InsightGenerator.generate_task_insight(task, "Fault"),
    "model": InsightGenerator.generate_model_insight(best, task, results["all_results"][best]),
}
adapted = ExpertiseAdapter.adapt_insights(insights, "beginner")
print("Beginner insight:", adapted["task"][:100])

print()
print("=== FULL SMOKE TEST PASSED ===")
