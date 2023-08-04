from typing import Tuple, TypedDict


class RecognitionPairInput(TypedDict):
    sentence: str
    pos: int


RecognitionPairOutput = str

RecognitionPair = Tuple[RecognitionPairInput, RecognitionPairOutput]


class BaseRecognitionModel:

    def recognize(self, sentence_pos_tuples: 'list[tuple[str, int]]') -> list[RecognitionPairOutput]:
        raise NotImplementedError('BaseRecognitionModel is an abstract class, please use a subclass instead.')
