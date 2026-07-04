from .engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """Thin user-facing facade over the engine.

    Separating LLM from LLMEngine keeps the public API stable while
    allowing the engine internals to evolve independently.
    """

    pass
