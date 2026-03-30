import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from argparse import ArgumentParser


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro"),
        "f1": f1_score(labels, predictions, average="macro"),
    }

class RelevancyClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data.to_dict(orient='records')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["atomic_fact"]
        label = self.data[idx]["label"]
        label = 1 if label=='relevant' else 0

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pretrained_model')
    parser.add_argument('--trg_lang')
    parser.add_argument('--output_model')
    
    args = parser.parse_args()
    
    trg_lang = args.trg_lang
    output_model = args.output_model
    pretrained_model = args.pretrained_model
    
    main_dir = Path(f'data/multi-wikimedcare/relevancy_ds/{trg_lang}')
    output_dir = f'data/multi-wikimedcare/model/{output_model}'
    
    train_df = pd.read_csv(main_dir/'train.csv')
    dev_df = pd.read_csv(main_dir/'dev.csv')
    
    print('Train data')
    print(train_df.groupby(['label'])['atomic_fact'].count())
    
    print('Dev data')
    print(dev_df.groupby(['label'])['atomic_fact'].count())
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    train_dataset = RelevancyClassificationDataset(train_df, tokenizer)
    dev_dataset = RelevancyClassificationDataset(dev_df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model,
    num_labels=train_df['label'].nunique()
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",  # fixed name: was eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",  # moved here
        greater_is_better=True,           # moved here
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,              # fixed name
        compute_metrics=compute_metrics,
    )

    trainer.train()