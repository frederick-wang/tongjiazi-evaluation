import re
from typing import List, cast
import tqdm
from transformers import GPT2LMHeadModel, BertTokenizer
from .BaseRecognitionModel import BaseRecognitionModel, RecognitionPairOutput
import torch


class GPT2RecognitionModel(BaseRecognitionModel):
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

    def __recognize(self, sentence: str, pos: int) -> str:
        c = sentence[pos]
        if c not in self.confusion_dict:
            return c
        confused_chars = self.confusion_dict[c]
        min_confused_perplexity = self.__calc_perplexity(sentence)
        min_confused_char = c
        for confused_char in confused_chars:
            confused_sentence = sentence[:pos] + confused_char + sentence[pos + 1:]
            confused_perplexity = self.__calc_perplexity(confused_sentence)
            if confused_perplexity < min_confused_perplexity:
                min_confused_perplexity = confused_perplexity
                min_confused_char = confused_char
        return min_confused_char

    def recognize(self, sentence_pos_tuples: 'list[tuple[str, int]]') -> list[RecognitionPairOutput]:
        return [
            self.__recognize(sentence, pos) for sentence, pos in tqdm.tqdm(sentence_pos_tuples, desc='使用 GPT2 识别句集正字中')
        ]
