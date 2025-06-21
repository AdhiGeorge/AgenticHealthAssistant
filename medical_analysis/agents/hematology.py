from .base import MedicalAgent
# from langgraph.graph import StateGraph

class HematologyAgent(MedicalAgent):
    def __init__(self, model: str = "gemini-pro", use_gpu: bool = True):
        super().__init__(specialty="hematology", model=model, use_gpu=use_gpu)

    def analyze(self, content: str, use_cache: bool = True):
        prompt = self.prompts.get("analysis", "").format(content=content[:10000])
        cot_prompt = (
            f"Let's think step by step as a hematologist.\n"
            f"{prompt}\n"
            f"Step 1: Identify blood count and coagulation findings.\n"
            f"Step 2: Review bone marrow and transfusion history.\n"
            f"Step 3: Assess related conditions.\n"
            f"Step 4: Formulate recommendations.\n"
            f"Step 5: Highlight critical issues.\n"
        )
        return {"prompt": cot_prompt, "specialty": self.specialty}

    def answer_query(self, query: str, context: dict):
        prompt = self.prompts.get("query", "").format(query=query)
        cot_prompt = (
            f"Let's think step by step as a hematologist.\n"
            f"{prompt}\n"
            f"Step 1: Understand the question.\n"
            f"Step 2: Retrieve relevant context from the report.\n"
            f"Step 3: Apply hematology knowledge.\n"
            f"Step 4: Formulate a clear answer.\n"
        )
        return {"prompt": cot_prompt, "specialty": self.specialty} 