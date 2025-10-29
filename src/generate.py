import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt="girl, you make me feel", model_dir="models/mj-gpt2"):
    # Загружаем токенайзер и модель
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Устанавливаем pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Перенос модели на CPU для стабильности
    device = "cpu"
    model.to(device)

    # Токенизация prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Генерация по шагам с проверкой nan/inf
    max_length = 100
    generated = input_ids
    for _ in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]

            # Стабилизируем вероятности
            next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=1e9, neginf=-1e9)
            probs = torch.softmax(next_token_logits, dim=-1)

            # Сэмплинг
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("Generated text:\n")
    print(text)

if __name__ == "__main__":
    generate_text()
