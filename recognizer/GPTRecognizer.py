import re
from typing import Iterable, cast

import torch
from torch import Tensor
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from loader import confusion_dict
from recognizer.BaseRecognizer import BaseRecognizer

# logging.set_verbosity_error()


class BertMLMRecognizer(BaseRecognizer):
    '''Bert MLM 语言模型通假字识别器'''

    __model_path: str
    __device: torch.device
    __confusion_dict: 'dict[str, set[str]]'
    __model: BertForMaskedLM
    __tokenizer: BertTokenizerFast
    __config: BertConfig
    __num_embeddings: int
    batch_size: int = 32

    def __init__(self, model_path: str, *, device: 'str | None' = None):
        self.__model_path = model_path
        if device is None:
            self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.__device = torch.device(device)
        self.__model = BertForMaskedLM.from_pretrained(self.__model_path).to(self.__device)  # type: ignore
        self.__tokenizer = BertTokenizerFast.from_pretrained(self.__model_path)
        self.__config = BertConfig.from_pretrained(self.__model_path)  # type: ignore
        self.__num_embeddings = self.__model.get_input_embeddings().num_embeddings  # type: ignore
        self.__confusion_dict = confusion_dict

    @property
    def num_embeddings(self) -> int:
        return self.__num_embeddings

    def recognize(self, sentences: 'Iterable[str]') -> 'Iterable[list[tuple[str, str, int]]]':
        return (self.__recognize(sentence) for sentence in sentences)

    def __recognize(
        self,
        sentence: str,
        *,
        separators: str = r'，。！？；：…—,.!?;:-',
    ) -> 'list[tuple[str, str, int]]':
        if self.__model is None or self.__tokenizer is None:
            raise RuntimeError('Model or tokenizer is not initialized.')
        result: 'list[tuple[str, str, int]]' = []  # list of (tongjiazi, benzi, position)
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
                       if char in self.__confusion_dict]
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
            ).to(self.__device)
            for row_i, (_, _, c, _) in enumerate(batch_frag_pos_char_triples):
                confusion_set = self.__confusion_dict[c] | {c}
                confusion_indices = cast(
                    'list[int]',
                    self.__tokenizer.convert_tokens_to_ids(list(confusion_set)),
                )
                confusion_mask[row_i, confusion_indices] = False
            batch_input_ids: Tensor = self.__tokenizer(
                batch_masked_sentences,
                return_tensors='pt',
                padding=True,
            ).input_ids.to(self.__device)
            mask_token_indices = torch.where(batch_input_ids == self.__tokenizer.mask_token_id)
            token_logits = self.__model(batch_input_ids).logits
            batch_mask_token_logits = token_logits[mask_token_indices].masked_fill(
                confusion_mask,
                -float('inf'),
            )
            batch_max_confusion_char_indices = torch.argmax(batch_mask_token_logits, dim=1)
            batch_confusion_chars = cast(
                'list[str]',
                self.__tokenizer.convert_ids_to_tokens(batch_max_confusion_char_indices.tolist()),
            )
            result.extend((c, confusion_char, pos)
                          for confusion_char, (_, pos, c, _) in zip(batch_confusion_chars, batch_frag_pos_char_triples)
                          if c != confusion_char and confusion_char in self.__confusion_dict[c])
        return result


if __name__ == '__main__':
    recognizer = BertMLMRecognizer(model_path='/media/disk1/public/bert-pt31')
    test_case = [
        '故其在大譴大何之域者，聞譴何則白冠氂纓，盤水加劍，造請室而辠耳。',
        '兩界耕桑交跡，吏不何問。',
        '又下令不何止夜行，使民自便，境内以安。',
    ]
    for case in test_case:
        print(recognizer.__recognize(case))
