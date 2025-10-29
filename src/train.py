import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def train_model():
    dataset_path = "data/mj_dataset"
    model_name = "gpt2"
    output_dir = "models/mj-gpt2"
    log_dir = "runs/mj-gpt2"
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)

    print("Загружаем датасет...")
    ds = load_from_disk(dataset_path)

    print("Загружаем токенайзер...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["train"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    print("Токенизация...")
    tokenized_ds = ds.map(tokenize, batched=True, remove_columns=["train"])

    print("Загружаем модель...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === Настройки обучения ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to=["tensorboard"],
        logging_dir=log_dir,
        learning_rate=2e-5,  
        warmup_steps=100,
        max_grad_norm=1.0,   # gradient clipping
        fp16=False,          # отключаем для стабильности на MPS
        label_smoothing_factor=0.1,
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

    # === Проверим несколько батчей ===
    print("Проверяем несколько батчей лосса...")
    logs = trainer.state.log_history[-10:]
    for log in logs:
        if "loss" in log:
            print(log)

    # === Сохранение модели ===
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Модель сохранена в {output_dir}")

    # === Визуализация ===
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        print("Создаём графики обучения...")
        ea = EventAccumulator(log_dir)
        ea.Reload()

        metrics = {}
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            metrics[tag] = pd.DataFrame({"step": steps, "value": values})

            plt.figure(figsize=(8, 4))
            plt.plot(steps, values, label=tag)
            plt.title(f"{tag}")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            img_path = Path(images_dir) / f"{tag.replace('/', '_')}.png"
            plt.savefig(img_path)
            plt.close()

        print(f"Графики сохранены в {images_dir}/")

    except Exception as e:
        print("Не удалось построить графики:", e)


if __name__ == "__main__":
    train_model()
