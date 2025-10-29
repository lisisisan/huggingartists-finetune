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
    dataset_path = "data/mj_dataset"
    model_name = "gpt2"
    output_dir = "models/mj-gpt2"
    log_dir = "runs/mj-gpt2"

    print("📦 Загружаем датасет...")
    ds = load_from_disk(dataset_path)

    print("🔠 Загружаем токенайзер...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["train"], truncation=True, padding="max_length", max_length=128)

    print("🧩 Токенизация...")
    tokenized_ds = ds.map(tokenize, batched=True, remove_columns=["train"])

    print("🤖 Загружаем модель...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ✅ Совместимая версия TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_strategy="epoch",  # можно оставить — всё ещё поддерживается
        learning_rate=5e-5,
        warmup_steps=50,
        report_to="tensorboard",  # логирование в TensorBoard
        logging_dir=log_dir,
        logging_steps=50,
        fp16=False,  # fp16 нельзя использовать на MPS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
    )

    print("🚀 Начинаем обучение...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Модель сохранена в {output_dir}")

if __name__ == "__main__":
    train_model()
