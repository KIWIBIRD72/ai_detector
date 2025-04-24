# 🚀 Project start
Python 3.12.4

- Активация локального окружения
  ```bash
  # Windows:
  venv\Scripts\activate
  # macOS / Linux:
  source venv/bin/activate
  ```
- Тренировка модели и выгрузка модели в статические файлы
  ```bash
  python3 app/services/train_service.py
  ```

- Установить зависимости из requirements.txt
  ```bash
  pip install -r requirements.txt
  ```

- dev mode
    ```bash
     fastapi dev main.py
    ```
- production mode
    ```bash
    fastapi run    
    ```
