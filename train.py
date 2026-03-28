import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

import os
import pickle
if not os.path.exists("model"):
    os.makedirs("model")



# تحميل البيانات
df1 = pd.read_excel("data/ISIC04-2014.xls")
df2 = pd.read_excel("data/feedback.xlsx")

df1 = df1.rename(columns={"description": "text", "code": "label"})
df2 = df2.rename(columns={"text": "text", "code": "label"})

df = pd.concat([df1[["text", "label"]], df2], ignore_index=True).dropna()

# encoding
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# حفظ encoder
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# dataset
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

def tokenize(x):
    tokenized = tokenizer(x["text"], truncation=True, padding="max_length")
    tokenized["labels"] = x["label_encoded"]
    return tokenized

dataset = dataset.map(tokenize)
dataset = dataset.train_test_split(test_size=0.1)

# model
model = AutoModelForSequenceClassification.from_pretrained(
    "aubmindlab/bert-base-arabertv02",
    num_labels=len(df["label_encoded"].unique())
)

training_args = TrainingArguments(
    output_dir="model/trained_model",
    num_train_epochs=3,
    per_device_train_batch_size=8
)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()

# حفظ النموذج
model.save_pretrained("model/trained_model")

# 🔥 مهم جداً: حفظ tokenizer
tokenizer.save_pretrained("model/trained_model")