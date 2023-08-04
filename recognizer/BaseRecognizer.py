from typing import Iterable


class BaseRecognizer:
    '''通假字识别器基类'''

    def recognize(self, sentences: 'Iterable[str]') -> 'Iterable[list[tuple[str, str, int]]]':
        """
        识别句子中的通假字。
        
        Args:
            sentences(Iterable[str]): 待识别的句子
        
        Returns:
            Iterable[list[tuple[str, str, int]]]: 识别结果，每个元素为（通假字，正字，位置）
        """
        raise NotImplementedError('recognize() not implemented, please use a subclass of BaseRecognizer')

    def __call__(self, sentences: 'Iterable[str]') -> 'Iterable[list[tuple[str, str, int]]]':
        return self.recognize(sentences)

    def recognize_sentence(self, sentence: str) -> 'list[tuple[str, str, int]]':
        """
        识别句子中的通假字。
        
        Args:
            sentence(str): 待识别的句子
        
        Returns:
            list[tuple[str, str, int]]: 识别结果，每个元素为（通假字，正字，位置）
        """
        return iter(self.recognize([sentence])).__next__()

    def recognize_to_list(self, sentences: 'Iterable[str]') -> 'list[list[tuple[str, str, int]]]':
        """
        识别句子中的通假字。
        
        Args:
            sentences(Iterable[str]): 待识别的句子
        
        Returns:
            list[list[tuple[str, str, int]]]: 识别结果，每个元素为（通假字，正字，位置）
        """
        return list(self.recognize(sentences))
