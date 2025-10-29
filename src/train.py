import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from pathlib import Path

def train_model():
    # === Настройки ===
    dataset_path = "data/mj_dataset"
    model_name = "gpt2"  # можно заменить на distilgpt2 для ускорения
    output_dir = "models/mj-gpt2"
    log_dir = "runs/mj-gpt2"

    # === Загрузка данных ===
    print("Загружаем датасет...")
    ds = load_from_disk(dataset_path)

    # === Загрузка токенайзера ===
    print("Загружаем токенайзер...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 не имеет pad_token

    # === Токенизация ===
    def tokenize(batch):
        return tokenizer(batch["train"], truncation=True, padding="max_length", max_length=128)

    print("Токенизация...")
    tokenized_ds = ds.map(tokenize, batched=True, remove_columns=["train"])

    # === Модель ===
    print("Загружаем модель...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # === Data collator ===
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === Обучение ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to=["tensorboard"],
        logging_dir=log_dir,
        learning_rate=5e-5,
        warmup_steps=50,
        fp16=torch.backends.mps.is_available(),  # для Mac GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
    )

    print("Начинаем обучение...")
    trainer.train()

    # === Сохранение ===
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Модель сохранена в {output_dir}")

if __name__ == "__main__":
    train_model()
