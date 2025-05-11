from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
import evaluate

# 1. Load and filter SNLI
snli = load_dataset("snli")
snli = snli.filter(lambda ex: ex["label"] != -1)

# 2. Tokenizer & Preprocessing
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

# 3. Format for PyTorch
snli.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4. Load DistilBERT for 3-way classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
)

# 5. TrainingArguments with fp16 for speed
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

# 6. Metrics
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels)


# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=snli["train"],
    eval_dataset=snli["validation"],
    compute_metrics=compute_metrics,
)

# 8. Train & Evaluate
trainer.train()
results = trainer.evaluate(snli["test"])
print("Test results:", results)

trainer.save_model("./data/models/")
tokenizer.save_pretrained("./data/models/")
