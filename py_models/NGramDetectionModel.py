import re
from typing import List
import kenlm
import tqdm
from .BaseDetectionModel import BaseDetectionModel, DetectionPairOutput


class NGramDetectionModel(BaseDetectionModel):
    model_path: str
    model: kenlm.Model
    confusion_dict: 'dict[str, set[str]]'

    def __init__(self, model_path: str, confusion_dict: 'dict[str, set[str]]'):
        self.model_path = model_path
        self.model = kenlm.Model(model_path)
        self.confusion_dict = confusion_dict

    def __calc_perplexity(self, fragment: str) -> float:
        """
        计算句子片段的困惑度（无标点）
        
        Args:
            fragment(str): 句子片段，不能包含标点符号
        
        Returns:
            float: 困惑度，越小越好
        """
        fragment = ' '.join(re.sub(r'\s+', '', fragment))
        N = len(fragment.split())
        log10_p = self.model.score(fragment, bos=False, eos=False)
        # p ** (-1.0 / N) = (10 ** log10_p) ** (-1.0 / N) = 10 ** (-log10_p / N)
        return 10**(-log10_p / N)

    def __detect(self, sentence: str) -> 'list[int]':
        # 按照标点将句子分割成多个句子片段
        separators = r'，。！？；：…—,.!?;:-'
        fragments = filter(None, re.split(rf'([{separators}])', sentence))
        result: 'list[int]' = []
        fragment_start_index = 0
        for fragment in fragments:
            if fragment in separators:
                continue
            perplexity = self.__calc_perplexity(fragment)
            for i, c in enumerate(fragment):
                if c not in self.confusion_dict:
                    continue
                confusion_set = self.confusion_dict[c]
                for confusion_char in confusion_set:
                    confusion_sentence = fragment[:i] + confusion_char + fragment[i + 1:]
                    confusion_perplexity = self.__calc_perplexity(confusion_sentence)
                    if confusion_perplexity < perplexity:
                        result.append(fragment_start_index + i)
            fragment_start_index += len(fragment)
        return result

    def detect(self, sentences: 'List[str]') -> List[DetectionPairOutput]:
        return [self.__detect(x) for x in tqdm.tqdm(sentences, desc='使用 N-Gram 检测句集通假字中')]
