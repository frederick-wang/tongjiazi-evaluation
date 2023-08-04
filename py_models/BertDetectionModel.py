import re
from typing import List, cast
import tqdm
import torch
from torch import Tensor
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from .BaseDetectionModel import BaseDetectionModel, DetectionPairOutput


class BertDetectionModel(BaseDetectionModel):
    model_path: str
    model: BertForMaskedLM
    tokenizer: BertTokenizerFast
    confusion_dict: 'dict[str, set[str]]'
    device: torch.device
    num_embeddings: int
    batch_size: int = 32

    def __init__(self, model_path: str, confusion_dict: 'dict[str, set[str]]'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = BertForMaskedLM.from_pretrained(self.model_path).to(self.device)  # type: ignore
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        self.num_embeddings = self.model.get_input_embeddings().num_embeddings  # type: ignore
        self.confusion_dict = confusion_dict

    def __detect(self, sentence: str) -> 'List[int]':
        separators = r'，。！？；：…—,.!?;:-'
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Model or tokenizer is not initialized.')
        result: 'list[int]' = []  # list of position
        # Split the sentence into fragments by punctuation, the punctuations are also included in the fragments.
        fragments: 'list[str]' = [frag for frag in re.split(rf'([{separators}])', sentence) if frag]
        frag_start_idx = 0
        frag_start_idx_list: 'list[int]' = []
        for fragment in fragments:
            frag_start_idx_list.append(frag_start_idx)
            frag_start_idx += len(fragment)
        used_fragments = (
            (frag, start_idx) for frag, start_idx in zip(fragments, frag_start_idx_list) if frag[0] not in separators)
        frag_pos_char_quads: 'list[tuple[str, int, str, int]]' = []  # (fragment, frag_pos, char, char_pos)
        for fragment, frat_start_idx in used_fragments:
            triples = [(fragment, frat_start_idx + char_pos, char, char_pos)
                       for char_pos, char in enumerate(fragment)
                       if char in self.confusion_dict]
            frag_pos_char_quads.extend(triples)
        masked_sentences = [
            f'{frag[:char_pos]}[MASK]{frag[char_pos + 1:]}' for frag, _, _, char_pos in frag_pos_char_quads
        ]
        BATCH_SIZE = self.batch_size
        for i in range(0, len(masked_sentences), BATCH_SIZE):
            batch_masked_sentences = masked_sentences[i:i + BATCH_SIZE]
            batch_frag_pos_char_triples = frag_pos_char_quads[i:i + BATCH_SIZE]
            BATCH_ACTUAL_SIZE = len(batch_masked_sentences)
            confusion_mask = torch.ones(
                (BATCH_ACTUAL_SIZE, self.num_embeddings),
                dtype=torch.bool,
            ).to(self.device)
            for row_i, (_, _, c, _) in enumerate(batch_frag_pos_char_triples):
                confusion_set = self.confusion_dict[c] | {c}
                confusion_indices = cast(
                    'list[int]',
                    self.tokenizer.convert_tokens_to_ids(list(confusion_set)),
                )
                confusion_mask[row_i, confusion_indices] = False
            batch_input_ids: Tensor = self.tokenizer(
                batch_masked_sentences,
                return_tensors='pt',
                padding=True,
            ).input_ids.to(self.device)
            mask_token_indices = torch.where(batch_input_ids == self.tokenizer.mask_token_id)
            token_logits = self.model(batch_input_ids).logits
            batch_mask_token_logits = token_logits[mask_token_indices].masked_fill(
                confusion_mask,
                -float('inf'),
            )
            batch_max_confusion_char_indices = torch.argmax(batch_mask_token_logits, dim=1)
            batch_confusion_chars = cast(
                'list[str]',
                self.tokenizer.convert_ids_to_tokens(batch_max_confusion_char_indices.tolist()),
            )
            result.extend(pos for confusion_char, (_, pos, c,
                                                   _) in zip(batch_confusion_chars, batch_frag_pos_char_triples)
                          if c != confusion_char and confusion_char in self.confusion_dict[c])
        return result

    def detect(self, sentences: 'List[str]') -> List[DetectionPairOutput]:
        return [self.__detect(x) for x in tqdm.tqdm(sentences, desc='使用 Bert 检测句集通假字中')]
