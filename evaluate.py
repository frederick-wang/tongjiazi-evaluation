import time
from datetime import datetime
from io import TextIOWrapper
from typing import List, Tuple

from evaluator.DetectionEvaluator import DetectionEvaluator
from evaluator.RecognitionEvaluator import RecognitionEvaluator
from loader import (based_detection_pairs, based_recognition_pairs, confusion_dict, extended_detection_pairs,
                    extended_recognition_pairs)
from py_models.BaseDetectionModel import BaseDetectionModel, DetectionPair
from py_models.BaseRecognitionModel import (BaseRecognitionModel, RecognitionPair)
from py_models.GPT2RecognitionModel import GPT2RecognitionModel
from py_models.NGramDetectionModel import NGramDetectionModel
from py_models.NGramRecognitionModel import NGramRecognitionModel
from py_models.GPT2DetectionModel import GPT2DetectionModel
from py_models.BertDetectionModel import BertDetectionModel
from py_models.BertRecognitionModel import BertRecognitionModel
from py_models.TongjiaziDetectionBertDetectionModel import TongjiaziDetectionBertDetectionModel
from py_models.TongjiaziDetectionBertWithConfusionDetectionModel import TongjiaziDetectionBertWithConfusionDetectionModel

daizhige_trigram_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/ngram/daizhigev20_trigram.klm'  # 殆知阁 Tri-Gram 模型
daizhige_bigram_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/ngram/daizhigev20_bigram.klm'  # 殆知阁 Bi-Gram 模型
siku_bigram_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/ngram/siku_bigram.klm'  # SikuBigram 模型
siku_trigram_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/ngram/siku_trigram.klm'  # SikuTrigram 模型
daizhige_bert_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/bert/DaizhigeBert'  # DaizhigeBert 模型
siku_bert_model_path = 'SIKU-BERT/sikubert'  # SIKU-BERT 模型
daizhige_gpt2_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/gpt2/DaizhigeGPT2'  # DaizhigeGPT2 模型
siku_gpt2_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/gpt2/SikuGPT2'  # SikuGPT2 模型
tongjiazi_detection_daizhige_bert_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/tasks/TongjiaziDetectionBert/saved/model_lr_5e-05_epochs_5_batch_size_8_seed_42_test_size_0.1_20230411175827'  # TongjiaziDetectionDaizhigeBert 模型
tongjiazi_detection_siku_bert_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/tasks/TongjiaziDetectionSikuBert/saved/model_lr_5e-05_epochs_5_batch_size_8_seed_42_test_size_0.1_20230411175936'  # TongjiaziDetectionSikuBert 模型
zhengzi_recognition_daizhige_bert_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/tasks/ZhengziRecognitionBert/saved/model_lr_5e-05_epochs_5_batch_size_8_seed_42_test_size_0.1_20230411175423'  # ZhengziRecognitionDaizhigeBert 模型
zhengzi_recognition_siku_bert_model_path = '/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/tasks/ZhengziRecognitionSikuBert/saved/model_lr_5e-05_epochs_5_batch_size_8_seed_42_test_size_0.1_20230411175702'  # ZhengziRecognitionSikuBert 模型


def main():
    report_file_path = f'evaluation_report/report_{datetime.now().strftime("%Y%m%d%H%M%S")}.tsv'
    with open(report_file_path, 'w', encoding='utf-8') as f_report:
        f_report.write('模型\t任务\t精确率\t召回率\tF1\t耗时（秒）\n')
        start_time = time.time()
        model_list: List[Tuple[str, str]] = [
            (daizhige_bigram_model_path, 'DaizhigeBigram 模型'),
            (siku_bigram_model_path, 'SikuBigram 模型'),
            (daizhige_trigram_model_path, 'DaizhigeTrigram 模型'),
            (siku_trigram_model_path, 'SikuTrigram 模型'),
        ]
        for model_path, model_name in model_list:
            run_detection_evaluation(
                f_report,
                model=NGramDetectionModel(model_path=model_path, confusion_dict=confusion_dict),
                model_name=model_name,
                tasks=[
                    (based_detection_pairs, '通假字检测（基础）'),
                    (extended_detection_pairs, '通假字检测（拓展）'),
                ],
            )
            run_recognition_evaluation(
                f_report,
                model=NGramRecognitionModel(model_path=model_path, confusion_dict=confusion_dict),
                model_name=model_name,
                tasks=[
                    (based_recognition_pairs, '正字识别（基础）'),
                    (extended_recognition_pairs, '正字识别（拓展）'),
                ],
            )
        model_list = [
            (daizhige_gpt2_model_path, 'DaizhigeGPT2 模型'),
            (siku_gpt2_model_path, 'SikuGPT2 模型'),
        ]
        for model_path, model_name in model_list:
            run_detection_evaluation(
                f_report,
                model=GPT2DetectionModel(model_path=model_path, confusion_dict=confusion_dict),
                model_name=model_name,
                tasks=[
                    (based_detection_pairs, '通假字检测（基础）'),
                    (extended_detection_pairs, '通假字检测（拓展）'),
                ],
            )
            run_recognition_evaluation(
                f_report,
                model=GPT2RecognitionModel(model_path=model_path, confusion_dict=confusion_dict),
                model_name=model_name,
                tasks=[
                    (based_recognition_pairs, '正字识别（基础）'),
                    (extended_recognition_pairs, '正字识别（拓展）'),
                ],
            )
        model_list = [
            (daizhige_bert_model_path, 'DaizhigeBert 模型'),
            (siku_bert_model_path, 'SikuBert 模型'),
        ]
        for model_path, model_name in model_list:
            run_detection_evaluation(
                f_report,
                model=BertDetectionModel(model_path=model_path, confusion_dict=confusion_dict),
                model_name=model_name,
                tasks=[
                    (based_detection_pairs, '通假字检测（基础）'),
                    (extended_detection_pairs, '通假字检测（拓展）'),
                ],
            )
            run_recognition_evaluation(
                f_report,
                model=BertRecognitionModel(model_path=model_path, confusion_dict=confusion_dict),
                model_name=model_name,
                tasks=[
                    (based_recognition_pairs, '正字识别（基础）'),
                    (extended_recognition_pairs, '正字识别（拓展）'),
                ],
            )
        model_list = [
            (tongjiazi_detection_daizhige_bert_model_path, 'TongjiaziDetectionDaizhigeBert 模型'),
            (tongjiazi_detection_siku_bert_model_path, 'TongjiaziDetectionSikuBert 模型'),
        ]
        for model_path, model_name in model_list:
            run_detection_evaluation(
                f_report,
                model=TongjiaziDetectionBertWithConfusionDetectionModel(model_path=model_path,
                                                                        confusion_dict=confusion_dict),
                model_name=model_name,
                tasks=[
                    (based_detection_pairs, '通假字检测（基础）'),
                    (extended_detection_pairs, '通假字检测（拓展）'),
                ],
            )
        model_list = [
            (tongjiazi_detection_daizhige_bert_model_path, 'TongjiaziDetectionDaizhigeBert（无混淆集） 模型'),
            (tongjiazi_detection_siku_bert_model_path, 'TongjiaziDetectionSikuBert（无混淆集） 模型'),
        ]
        for model_path, model_name in model_list:
            run_detection_evaluation(
                f_report,
                model=TongjiaziDetectionBertDetectionModel(model_path=model_path),
                model_name=model_name,
                tasks=[
                    (based_detection_pairs, '通假字检测（基础）'),
                    (extended_detection_pairs, '通假字检测（拓展）'),
                ],
            )
        model_list = [
            (zhengzi_recognition_daizhige_bert_model_path, 'ZhengziRecognitionDaizhigeBert 模型'),
            (zhengzi_recognition_siku_bert_model_path, 'ZhengziRecognitionSikuBert 模型'),
        ]
        for model_path, model_name in model_list:
            run_recognition_evaluation(
                f_report,
                model=BertRecognitionModel(model_path=model_path, confusion_dict=confusion_dict),
                model_name=model_name,
                tasks=[
                    (based_recognition_pairs, '正字识别（基础）'),
                    (extended_recognition_pairs, '正字识别（拓展）'),
                ],
            )
        stop_time = time.time()
        print('\n' + f'总耗时：{stop_time-start_time:.2f} 秒')


def run_detection_evaluation(f_report: TextIOWrapper, model: BaseDetectionModel, model_name: str,
                             tasks: List[Tuple[List[DetectionPair], str]]):
    for pairs, task_name in tasks:
        t1 = time.time()
        evaluator = DetectionEvaluator(model=model, pairs=pairs, model_name=model_name, task_name=task_name)
        formatted_result = evaluator.get_formatted_result()
        t2 = time.time()
        record = f'{formatted_result}\t{t2-t1:.6f}'
        print(record)
        f_report.write(record + '\n')


def run_recognition_evaluation(f_report: TextIOWrapper, model: BaseRecognitionModel, model_name: str,
                               tasks: List[Tuple[List[RecognitionPair], str]]):
    for pairs, task_name in tasks:
        t1 = time.time()
        evaluator = RecognitionEvaluator(model=model, pairs=pairs, model_name=model_name, task_name=task_name)
        formatted_result = evaluator.get_formatted_result()
        t2 = time.time()
        record = f'{formatted_result}\t{t2-t1:.6f}'
        print(record)
        f_report.write(record + '\n')


if __name__ == '__main__':
    main()
