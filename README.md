## 1Основные разделы README

Пример:

````markdown
# Fine-tuning GPT-2 on Michael Jackson Lyrics

## Описание
Проект дообучает GPT-2 на текстах песен Майкла Джексона, чтобы генерировать тексты в его стиле.  
Датасет: HuggingArtists Michael Jackson (~1414 песен).  
Модель: GPT-2 (small), токенизатор — стандартный GPT-2.

## Структура проекта
- `src/dataset.py` — загрузка и подготовка датасета
- `src/train.py` — дообучение модели GPT-2
- `src/generate.py` — генерация текста в стиле артиста
- `data/` — локальные данные и сохранённый dataset
- `models/mj-gpt2/` — сохранённая модель после тренировки
- `runs/` — логи для TensorBoard

## Настройка
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

## Запуск

1. Подготовка датасета:

```bash
python src/dataset.py
```

2. Дообучение модели:

```bash
python src/train.py
```

3. Генерация текста:

```bash
python src/generate.py
```

## Результаты генерации

Пример с prompt `girl, you make me feel`:

```
girl, you make me feel Funkbreaklast1980 Tulbeen barking MVP Django resignedments...
```

Модель начала генерировать текст в стиле Майкла Джексона, но из-за малого объёма данных и короткой токенизации иногда появляются странные токены.

## Выводы по обучению

* Использование небольшого GPT-2 с ~1414 текстов песен даёт основу для генерации.
* Лучше увеличить `max_length` и размер батча для более связного текста.
* На Mac MPS иногда возникают `inf/nan` при генерации, решено генерацией по шагам и `torch.nan_to_num`.
* TensorBoard показывает метрики loss/train и loss/eval для каждой эпохи.

## Графики обучения

В TensorBoard:

```bash
tensorboard --logdir runs/mj-gpt2
```
