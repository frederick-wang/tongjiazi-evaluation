import re
from typing import Iterable

import kenlm

from loader import confusion_dict
from recognizer.BaseRecognizer import BaseRecognizer


class NGramRecognizer(BaseRecognizer):
    '''N-Gram 语言模型通假字识别器'''

    __model_path: str
    __lm: kenlm.Model
    __confusion_dict: 'dict[str, set[str]]'

    def __init__(self, model_path: str):
        self.__model_path = model_path
        self.__lm = kenlm.Model(self.__model_path)
        self.__confusion_dict = confusion_dict

    def recognize(self, sentences: 'Iterable[str]') -> 'Iterable[list[tuple[str, str, int]]]':
        return (self.__recognize(sentence) for sentence in sentences)

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
        log10_p = self.__lm.score(fragment, bos=False, eos=False)
        # p ** (-1.0 / N) = (10 ** log10_p) ** (-1.0 / N) = 10 ** (-log10_p / N)
        return 10**(-log10_p / N)

    def __recognize(self, sentence: str) -> 'list[tuple[str, str, int]]':
        # 按照标点将句子分割成多个句子片段
        separators = r'，。！？；：…—,.!?;:-'
        fragments = filter(None, re.split(rf'([{separators}])', sentence))
        result: 'list[tuple[str, str, int]]' = []
        fragment_start_index = 0
        for fragment in fragments:
            if fragment in separators:
                continue
            perplexity = self.__calc_perplexity(fragment)
            # print(f'{fragment=}, {perplexity=}')
            for i, c in enumerate(fragment):
                if c not in self.__confusion_dict:
                    continue
                confusion_set = self.__confusion_dict[c]
                for confusion_char in confusion_set:
                    confusion_sentence = fragment[:i] + confusion_char + fragment[i + 1:]
                    confusion_perplexity = self.__calc_perplexity(confusion_sentence)
                    # print(f'{c=}, {confusion_char=}, {confusion_sentence=}, {confusion_perplexity=}')
                    if confusion_perplexity < perplexity:
                        result.append((c, confusion_char, fragment_start_index + i))
            fragment_start_index += len(fragment)
        return result
