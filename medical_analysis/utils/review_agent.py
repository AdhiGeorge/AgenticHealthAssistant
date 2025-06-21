from medical_analysis.utils.logger import get_logger

class ReviewAgent:
    """Reflection agent to compare analysis to original data and score relevancy."""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.logger = get_logger("ReviewAgent")

    def score_relevancy(self, original, analysis):
        # Simple scoring: ratio of overlapping words (can be replaced with embedding similarity)
        original_words = set(original.lower().split())
        analysis_words = set(analysis.lower().split())
        overlap = len(original_words & analysis_words)
        score = overlap / max(len(original_words), 1)
        self.logger.info(f"Relevancy score: {score:.2f}")
        return score

    def review(self, original, analysis):
        score = self.score_relevancy(original, analysis)
        status = "approved" if score >= self.threshold else "retry"
        return status, score 