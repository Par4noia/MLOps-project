# MLOps Project (Ethiopian Insurance)

Проект реализует MLOps-пайплайн для задачи классификации страховых выплат на основе датасета Ethiopian Insurance.

## Структура

- `config.py` — конфигурация директорий и путей
- `run.py` — CLI для режима `update`, `inference`, `summary`
- `data/` — исходные и подготовленные данные
- `src/`:
  - `collection/` — чтение партионного датасета
  - `analysis/` — анализ качества, очистка и создание целевой метки
  - `preparation/` — препроцессинг и разделение по времени
  - `training/` — обучение моделей, валидация, сохранение версий
  - `validation/` — логика контроля качества и сравнения моделей
  - `serving/` — инференс модели по CSV
  - `summary/` — генерация отчета
- `models/versions/` — сохраненные версии моделей с метриками
- `reports/` — отчет по качеству и итоговый summary

## Как запустить

1. Убедитесь, что установлены зависимости:
```bash
pip install -r requirements.txt
```

2. Запустить сбор/обработку/обучение:
```bash
python run.py -mode update
```

3. Запустить инференс на тестовом файле:
```bash
python run.py -mode inference -file test_inference.csv
```

4. Получить сводный отчет:
```bash
python run.py -mode summary
```

## Контроль качества и версии модели

- В `src/training/trainer.py` реализована `cross_validate_model` (TimeSeriesSplit).
- В `src/validation/validator.py` реализована проверка метрик и выбор лучшей модели.
- Модель сохраняется в `models/versions/` с метриками в JSON.
