"""Deterministic scoring policy."""


class ScoringPolicy:
    def score(
        self,
        *,
        keyword_score: float,
        vector_score: float,
        is_public: bool,
        kind: str,
        exact_match: bool,
    ) -> float:
        score = (keyword_score * 0.65) + (vector_score * 0.35)
        if exact_match:
            score += 0.2
        if is_public:
            score += 0.05
        if kind in {"Function", "Class"}:
            score += 0.03
        return round(score, 6)
