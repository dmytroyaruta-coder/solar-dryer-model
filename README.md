# Solar Dryer — Streamlit Community Cloud

## Файли

- `streamlit_app.py` — основний застосунок;
- `requirements.txt` — залежності;
- `.streamlit/secrets.toml.example` — приклад налаштування Google Sheets.

## Розгортання

1. Створіть публічний репозиторій GitHub.
2. Завантажте `streamlit_app.py` і `requirements.txt`.
3. Відкрийте Streamlit Community Cloud.
4. Виберіть репозиторій, гілку `main` і файл `streamlit_app.py`.
5. У Advanced settings → Secrets додайте:

```toml
PRODUCTS_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/ВАШ_ID/pub?output=csv"
```

6. Натисніть Deploy.

Після розгортання застосунок матиме сталу адресу виду:

`https://назва-застосунку.streamlit.app`

Дані продуктів завантажуються з Google Sheets, погодні ряди — з NASA POWER API.
