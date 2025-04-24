from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")

train_data = dataset["train"].shuffle(seed=42)

split_ratio = 0.8
split_index = int(len(train_data) * split_ratio)

train_subset = train_data.select(range(split_index))
test_subset = train_data.select(range(split_index, len(train_data)))


def prepare_dataset(data):
    """
    Формирование новго датасета. Добавление лейблов
    """
    texts = []
    labels = []

    for example in data:
        texts.append(example["chosen"])  # Human текст
        labels.append(0)  # Метка 0 -> HUMAN

        texts.append(example["rejected"])  # AI текст
        labels.append(1)  # Метка 1 -> AI

    return Dataset.from_dict({"text": texts, "label": labels})


train_dataset = prepare_dataset(train_subset)
test_dataset = prepare_dataset(test_subset)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def preprocess(data):
    return tokenizer(data["text"], truncation=True, padding="max_length", max_length=128)


train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    bf16=True,
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=False,
    max_steps=200,
)

print('Start Trainer')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
