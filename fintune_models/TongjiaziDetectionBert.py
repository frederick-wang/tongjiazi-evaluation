import datetime
import json
import os
import random
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import tqdm
from transformers import BertForTokenClassification, BertTokenizerFast, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split


class TongjiaziDetectionDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item["sentence"]
        data = item["data"]

        tokenized = self.tokenizer(sentence, return_offsets_mapping=True)
        input_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        labels = torch.zeros(len(input_ids), dtype=torch.int8)

        for x in data:
            pos = x["pos"]
            for i, offset in enumerate(offsets):
                if offset[0] <= pos < offset[1]:
                    labels[i] = 1
                    assert sentence[pos] == tokens[i]
                    break

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": labels,
        }


def evaluate(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_dataloader)


def train(model, train_dataloader, val_dataloader, device, optimizer, epochs):
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        stop_time = time.time()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Duration: {stop_time - start_time:.2f} seconds")

        # Evaluate the model on the validation set
        val_loss = evaluate(model, val_dataloader, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the model after each epoch
        # model.save_pretrained(
        #     f"/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/tasks/TongjiaziDetectionBert/saved/model_epoch_{epoch + 1}"
        # )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    learning_rate = 5e-5
    epochs = 5
    seed = 42
    test_size = 0.1

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = BertForTokenClassification.from_pretrained(
        "/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/bert/DaizhigeBert",
        num_labels=2).to(device)  # type: ignore
    tokenizer = BertTokenizerFast.from_pretrained(
        "/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/default/bert/DaizhigeBert")

    # Load data function
    def load_data(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    train_jsonl = load_data("/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/data/raw/train.jsonl")
    # Split the data into training and validation sets
    train_data, val_data = train_test_split(train_jsonl, test_size=test_size, random_state=seed)

    train_dataset = TongjiaziDetectionDataset(train_data, tokenizer)
    val_dataset = TongjiaziDetectionDataset(val_data, tokenizer)

    # Define batch_size and default_data_collator
    default_data_collator = DataCollatorForTokenClassification(tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model_save_dir = "/media/disk2/wangzhaoji/Projects/tongjiazi-dataset-baseline/models/tasks/TongjiaziDetectionBert/saved"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    time_sign = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # record the training parameters to the model name
    postfix = f"lr_{learning_rate}_epochs_{epochs}_batch_size_{batch_size}_seed_{seed}_test_size_{test_size}_{time_sign}"
    model_save_path = f"{model_save_dir}/model_{postfix}"

    train(model, train_dataloader, val_dataloader, device, optimizer, epochs)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == '__main__':
    main()
