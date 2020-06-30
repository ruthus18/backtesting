# Бот для сбора торговой статистики для Binance

## Локальная установка

- Создать виртуальное окружение и установить зависимости:

    ```bash
    python -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
    ```

- Создать файл с переменными окружения:

    ```bash
    cp .env.example .env
    ```

- Для выгрузки графиков в .png файлы дополнительно потребуется установить orca: https://github.com/plotly/orca

- В переменных окружениях необходимо указать `BINANCE_API_KEY` и `BINANCE_API_SECRET` (ключи доступа до API Binance)

## Запуск

- Запуск статистики по валютной паре BTC-USDT за 18-20 год:

    ```bash
    python backtest.py
    ```