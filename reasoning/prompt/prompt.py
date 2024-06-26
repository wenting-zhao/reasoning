from abc import ABC, abstractmethod


class Prompt(ABC):
    def __init__(self, use_fewshot, k=4):
        self.use_fewshot = use_fewshot
        if self.use_fewshot:
            self.fewshot_examples = self.get_fewshot_examples(k)

    @abstractmethod
    def get_fewshot_examples(self, k):
        pass

    @abstractmethod
    def render(self, example):
        pass
