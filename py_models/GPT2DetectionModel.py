import re
from typing import List, cast
import tqdm
from transformers import GPT2LMHeadModel, BertTokenizer
from .BaseDetectionModel import BaseDetectionModel, DetectionPairOutput
import torch


class GPT2DetectionModel(BaseDetectionModel):
    model_path: str
    model: GPT2LMHeadModel
    tokenizer: BertTokenizer
    confusion_dict: 'dict[str, set[str]]'
    device: torch.device

    def __init__(self, model_path: str, confusion_dict: 'dict[str, set[str]]'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = cast(GPT2LMHeadModel, GPT2LMHeadModel.from_pretrained(model_path)).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
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
        input_ids = self.tokenizer.encode(fragment, return_tensors="pt").to(self.device)  # type: ignore
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    def __detect(self, sentence: str) -> 'List[int]':
        separators = r'，。！？；：…—,.!?;:-'
        fragments = filter(None, re.split(rf'([{separators}])', sentence))
        result: 'List[int]' = []
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
        return [self.__detect(x) for x in tqdm.tqdm(sentences, desc='使用 GPT2 检测句集通假字中')]
