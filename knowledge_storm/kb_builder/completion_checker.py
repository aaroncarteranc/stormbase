import dspy
from ..dataclass import KnowledgeBase


class ArticleSufficiencyCheck(dspy.Signature):
    """Assess whether a knowledge base has sufficient coverage to write a complete, well-sourced article.
    Answer 'yes' only if the structure shows broad coverage across multiple distinct subtopics.
    Answer 'no' if important aspects appear missing or coverage is shallow.
    """

    topic = dspy.InputField(prefix="Article topic: ", format=str)
    kb_structure = dspy.InputField(prefix="Knowledge base structure:\n", format=str)
    answer = dspy.OutputField(
        prefix="Sufficient coverage? Answer 'yes' or 'no':\n", format=str
    )


class CompletionChecker:
    def __init__(
        self,
        lm,
        min_floor: int = 10,
        check_interval: int = 5,
        max_ceiling: int = 40,
    ):
        self.lm = lm
        self.min_floor = min_floor
        self.check_interval = check_interval
        self.max_ceiling = max_ceiling
        self._checker = dspy.Predict(ArticleSufficiencyCheck)

    def should_check(self, turn_number: int) -> bool:
        if turn_number < self.min_floor:
            return False
        return (turn_number - self.min_floor) % self.check_interval == 0

    def is_sufficient(self, knowledge_base: KnowledgeBase, article_title: str) -> bool:
        structure = knowledge_base.get_node_hierarchy_string(
            include_indent=False,
            include_full_path=False,
            include_hash_tag=True,
            include_node_content_count=False,
        )
        with dspy.settings.context(lm=self.lm, show_guidelines=False):
            result = self._checker(topic=article_title, kb_structure=structure)
        return result.answer.strip().lower().startswith("yes")
