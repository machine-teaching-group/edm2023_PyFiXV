# A class for computing metrics of explanation
class ExplanationMetricsClerk:
    def __init__(self):
        self.n_generated_explanation = 0

    def add(
        self,
        final_generated_explanation: str,
    ):
        if final_generated_explanation is not None:
            self.n_generated_explanation += 1

        info = {
            "final_generated_explanation": final_generated_explanation,
        }

        return info

    def get_summary(self):
        summary = {
            "n": self.n_generated_explanation,
        }

        return summary
