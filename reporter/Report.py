from typing import TypedDict


class CharLevelReport(TypedDict):
    precision: float
    '''Precision, TP / (TP + FP)'''
    recall: float
    '''Recall, TP / (TP + FN)'''
    f1: float
    '''F1, 2 * Precision * Recall / (Precision + Recall)'''
    correct: int
    '''Correct, TP (True Positive)'''
    precision_count: int
    '''TP + FP (True Positive + False Positive)'''
    recall_count: int
    '''TP + FN (True Positive + False Negative)'''


class SentenceLevelReport(TypedDict):
    at_least_one_correct: float
    '''At least one correct, at least one correct sentence count / sentence count'''
    totally_correct: float
    '''Totally correct, totally correct sentence count / sentence count'''
    at_least_one_correct_count: int
    '''At least one correct sentence count'''
    totally_correct_count: int
    '''Totally correct sentence count'''
    sentence_count: int
    '''Sentence count'''


class Report(TypedDict):
    recognizer: str
    timestamp: float
    time: str
    cost_time: float
    char_level: CharLevelReport
    '''Character level report'''
    sentence_level: SentenceLevelReport
    '''Sentence level report'''
