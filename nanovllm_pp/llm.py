from .engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """Public API for the inference engine.

    LLMEngine handles all runtime logic. LLM provides a stable
    interface that can be versioned independently.
    """

    pass
