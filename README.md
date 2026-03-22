# MLOps MVP (Ethiopian Insurance)

Простой проект MVP для курса MLOps (задание 1) по обработке потоковых табличных данных.

## Запуск

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Поместите `insurance.csv` в папку `data/`.

3. Выполните обновление данных:

```bash
python run.py -mode update
```

Это создаст `data/raw/insurance_raw.csv`.

## Структура

- `run.py` - интерфейс командной строки.
- `src/collection/collector.py` - чтение и запись батчами.
- `config.py` - пути к каталогам.

## Следующие шаги

Будем добавлять модули подготовки, обучения, валидации и инференса.
