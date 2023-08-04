from datetime import datetime
import json
from py_models.BaseRecognitionModel import BaseRecognitionModel, RecognitionPair
from typing import List, Tuple

Unit = Tuple[str, int, str]  #  句子, 位置, 正字


class RecognitionEvaluator:
    model: BaseRecognitionModel
    pairs: List[RecognitionPair]
    model_name: str
    task_name: str

    def __init__(self, model: BaseRecognitionModel, pairs: List[RecognitionPair], model_name: str, task_name: str):
        self.model = model
        self.pairs = pairs
        self.model_name = model_name
        self.task_name = task_name

    def __get_result_tuple_set(self) -> 'set[Unit]':
        return {(pair[0]['sentence'], pair[0]['pos'], pair[1]) for pair in self.pairs}

    def __get_predicted_tuple_set(self) -> 'set[Unit]':
        sentence_pos_tuples = [(pair[0]['sentence'], pair[0]['pos']) for pair in self.pairs]
        predicted_result_list = self.model.recognize(sentence_pos_tuples=sentence_pos_tuples)
        return {(sent, pos, zhengzi) for (sent, pos), zhengzi in zip(sentence_pos_tuples, predicted_result_list)}

    def __get_metrics(self, result_tuple_set: 'set[Unit]',
                      predicted_tuple_set: 'set[Unit]') -> 'tuple[float, float, float]':
        true_positive = len(result_tuple_set & predicted_tuple_set)
        false_positive = len(predicted_tuple_set - result_tuple_set)
        false_negative = len(result_tuple_set - predicted_tuple_set)
        # 避免分母为 0
        if true_positive == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def get_formatted_result(self) -> str:
        precision, recall, f1 = self.get_result()
        return f'{self.model_name}\t{self.task_name}\t{precision:.2%}\t{recall:.2%}\t{f1:.4f}'

    def get_result(self) -> 'tuple[float, float, float]':
        result_tuple_set = self.__get_result_tuple_set()
        predicted_tuple_set = self.__get_predicted_tuple_set()
        self.export_details(result_tuple_set, predicted_tuple_set)
        precision, recall, f1 = self.__get_metrics(result_tuple_set, predicted_tuple_set)
        return precision, recall, f1

    def export_details(self, result_tuple_set: 'set[Unit]', predicted_tuple_set: 'set[Unit]'):
        true_positive = result_tuple_set & predicted_tuple_set
        false_positive = predicted_tuple_set - result_tuple_set
        false_negative = result_tuple_set - predicted_tuple_set
        # Convert to json
        true_positive = [{'sentence': sent, 'pos': pos, 'zhengzi': zhengzi} for sent, pos, zhengzi in true_positive]
        false_positive = [{'sentence': sent, 'pos': pos, 'zhengzi': zhengzi} for sent, pos, zhengzi in false_positive]
        false_negative = [{'sentence': sent, 'pos': pos, 'zhengzi': zhengzi} for sent, pos, zhengzi in false_negative]
        # Export
        json_data = {'true_positive': true_positive, 'false_positive': false_positive, 'false_negative': false_negative}
        postfix = datetime.now().strftime("%Y%m%d%H%M%S")
        details_file_path = f'evaluation_details/recognition_{self.model_name}_{self.task_name}_{postfix}.json'
        with open(details_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
