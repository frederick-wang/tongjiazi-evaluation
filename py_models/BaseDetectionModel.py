from typing import List, Tuple, TypedDict


class DetectionPairInput(TypedDict):
    sentence: str


DetectionPairOutput = List[int]

DetectionPair = Tuple[DetectionPairInput, DetectionPairOutput]


class BaseDetectionModel:

    def detect(self, sentences: 'list[str]') -> list[DetectionPairOutput]:
        raise NotImplementedError('BaseDectionModel is an abstract class, please use a subclass instead.')
