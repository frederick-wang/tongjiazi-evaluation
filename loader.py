import json

# 混淆集
with open('data/task/confusion_set/confusion.json', 'r') as f:
    confusion_dict_with_list: 'dict[str, list[str]]' = json.load(f)
confusion_dict: 'dict[str, set[str]]' = {key: set(value) for key, value in confusion_dict_with_list.items()}

# 通假字检测（基础）
with open('data/evaluation/based_detection_input.jsonl', 'r') as f_i:
    with open('data/evaluation/based_detection_output.jsonl', 'r') as f_o:
        input_items = [json.loads(line) for line in f_i if line]
        output_items = [json.loads(line) for line in f_o if line]
        based_detection_pairs = list(zip(input_items, output_items))

# 本字识别（基础）
with open('data/evaluation/based_recognition_input.jsonl', 'r') as f_i:
    with open('data/evaluation/based_recognition_output.jsonl', 'r') as f_o:
        input_items = [json.loads(line) for line in f_i if line]
        output_items = [json.loads(line) for line in f_o if line]
        based_recognition_pairs = list(zip(input_items, output_items))

# 通假字检测（拓展）
with open('data/evaluation/extended_detection_input.jsonl', 'r') as f_i:
    with open('data/evaluation/extended_detection_output.jsonl', 'r') as f_o:
        input_items = [json.loads(line) for line in f_i if line]
        output_items = [json.loads(line) for line in f_o if line]
        extended_detection_pairs = list(zip(input_items, output_items))

# 本字识别（拓展）
with open('data/evaluation/extended_recognition_input.jsonl', 'r') as f_i:
    with open('data/evaluation/extended_recognition_output.jsonl', 'r') as f_o:
        input_items = [json.loads(line) for line in f_i if line]
        output_items = [json.loads(line) for line in f_o if line]
        extended_recognition_pairs = list(zip(input_items, output_items))
