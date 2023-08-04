import json
import re
from typing import List, cast
import tqdm
import torch
from torch import Tensor
from transformers.models.bert.modeling_bert import BertForTokenClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from .BaseDetectionModel import BaseDetectionModel, DetectionPairOutput


class TongjiaziDetectionBertDetectionModel(BaseDetectionModel):
    model_path: str
    model: BertForTokenClassification
    tokenizer: BertTokenizerFast
    device: torch.device
    num_embeddings: int
    batch_size: int = 32

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = BertForTokenClassification.from_pretrained(self.model_path).to(self.device)  # type: ignore
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        self.num_embeddings = self.model.get_input_embeddings().num_embeddings  # type: ignore

    def __detect(self, sentence: str) -> 'List[int]':
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Model or tokenizer is not initialized.')
        self.model.eval()
        tokenized = self.tokenizer(
            sentence,
            return_tensors='pt',
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        offsets = tokenized["offset_mapping"][0].tolist()  # type: ignore
        del tokenized["offset_mapping"]
        logits = self.model(**tokenized).logits
        label_indices = torch.argmax(logits, dim=2)[0].tolist()
        pos_list: 'list[int]' = [offsets[i][0] for i, label in enumerate(label_indices) if label]
        # json_item = {'sentence': sentence, 'data': [{'tongjiazi': sentence[pos], 'pos': pos} for pos in pos_list]}
        # with open('test_output.jsonl', 'a') as f:
        #     f.write(json.dumps(json_item, ensure_ascii=False) + '\n')
        return pos_list

    def detect(self, sentences: 'List[str]') -> List[DetectionPairOutput]:
        return [self.__detect(x) for x in tqdm.tqdm(sentences, desc='使用 Bert 检测句集通假字中')]
