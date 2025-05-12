from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
import evaluate

snli = load_dataset("snli")
snli = snli.filter(lambda ex: ex["label"] != -1)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


snli = snli.map(preprocess, batched=True)

snli.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
)

training_args = TrainingArguments(
    output_dir="./distilbert-snli",
    eval_strategy="epoch",  # or eval_strategy="steps"
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=200,  # fewer warmup steps
    fp16=True,  # mixed-precision on T4
    save_strategy="epoch",
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=snli["train"],
    eval_dataset=snli["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate(snli["test"])
print("Test results:", results)

trainer.save_model("./data/models/")
tokenizer.save_pretrained("./data/models/")
