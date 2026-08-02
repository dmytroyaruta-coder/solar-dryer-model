from __future__ import annotations

import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from timezonefinder import TimezoneFinder


NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

NASA_PARAMETERS: dict[str, str] = {
    "ALLSKY_SFC_SW_DWN": "Сумарне короткохвильове випромінювання на горизонтальну поверхню",
    "ALLSKY_SFC_SW_DNI": "Пряма нормальна складова сонячного випромінювання",
    "ALLSKY_SFC_SW_DIFF": "Розсіяна складова сонячного випромінювання",
    "T2M": "Температура повітря на висоті 2 м",
    "RH2M": "Відносна вологість повітря на висоті 2 м",
    "T2MDEW": "Температура точки роси на висоті 2 м",
    "WS10M": "Швидкість вітру на висоті 10 м",
    "PS": "Атмосферний тиск біля поверхні",
}

REQUIRED_PRODUCT_COLUMNS = {
    "product_id",
    "product_name_uk",
    "product_form",
    "moisture_basis",
    "initial_moisture_pct",
    "final_moisture_pct",
    "target_drying_temp_c",
    "working_temp_min_c",
    "working_temp_max_c",
    "max_product_temp_c",
    "night_control_mode",
    "dew_point_margin_c",
    "data_status",
    "source_title",
    "source_url",
    "notes",
}

NUMERIC_PRODUCT_COLUMNS = [
    "initial_moisture_pct",
    "final_moisture_pct",
    "target_drying_temp_c",
    "working_temp_min_c",
    "working_temp_max_c",
    "max_product_temp_c",
    "dew_point_margin_c",
    "recommended_air_velocity_m_s",
    "layer_thickness_m",
    "specific_heat_dry_kj_kg_k",
    "bulk_density_kg_m3",
]


def get_products_sheet_url() -> str:
    """Читає URL Google Sheets зі Streamlit Secrets або змінної середовища."""
    try:
        url = str(st.secrets["PRODUCTS_SHEET_URL"]).strip()
    except (KeyError, FileNotFoundError):
        url = os.getenv("PRODUCTS_SHEET_URL", "").strip()

    if not url:
        raise RuntimeError(
            "Не задано PRODUCTS_SHEET_URL. Додайте його в Secrets "
            "застосунку Streamlit Community Cloud."
        )
    return url


def google_sheet_to_csv_url(sheet_url: str) -> str:
    """Перетворює публічне посилання Google Sheets на CSV URL."""
    sheet_url = sheet_url.strip()

    if "output=csv" in sheet_url or "export?format=csv" in sheet_url:
        return sheet_url

    if "/spreadsheets/d/e/" in sheet_url and "/pub" in sheet_url:
        separator = "&" if "?" in sheet_url else "?"
        return f"{sheet_url}{separator}output=csv"

    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", sheet_url)
    if not match:
        raise ValueError(
            "Не вдалося визначити ідентифікатор Google-таблиці."
        )

    sheet_id = match.group(1)
    parsed = urlparse(sheet_url)
    gid = (
        parse_qs(parsed.query).get("gid", [None])[0]
        or parse_qs(parsed.fragment).get("gid", [None])[0]
        or "0"
    )

    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={gid}"
    )


def validate_product_table(df: pd.DataFrame) -> pd.DataFrame:
    """Перевіряє структуру і логічну узгодженість довідника продуктів."""
    missing = REQUIRED_PRODUCT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "У Google-таблиці відсутні обов'язкові стовпці: "
            + ", ".join(sorted(missing))
        )

    result = df.copy()

    for column in NUMERIC_PRODUCT_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    if result["product_id"].duplicated().any():
        raise ValueError("Стовпець product_id містить повтори.")

    if result["product_name_uk"].duplicated().any():
        raise ValueError("Стовпець product_name_uk містить повтори.")

    invalid_moisture = (
        result["initial_moisture_pct"].isna()
        | result["final_moisture_pct"].isna()
        | (result["initial_moisture_pct"] <= 0)
        | (result["initial_moisture_pct"] >= 100)
        | (result["final_moisture_pct"] < 0)
        | (result["final_moisture_pct"] >= 100)
        | (
            result["initial_moisture_pct"]
            <= result["final_moisture_pct"]
        )
    )
    if invalid_moisture.any():
        bad = result.loc[
            invalid_moisture,
            "product_name_uk",
        ].tolist()
        raise ValueError(
            "Некоректні значення вологості для: "
            + ", ".join(map(str, bad))
        )

    invalid_temperature = (
        result["target_drying_temp_c"].isna()
        | result["working_temp_min_c"].isna()
        | result["working_temp_max_c"].isna()
        | (
            result["target_drying_temp_c"]
            < result["working_temp_min_c"]
        )
        | (
            result["target_drying_temp_c"]
            > result["working_temp_max_c"]
        )
    )
    if invalid_temperature.any():
        bad = result.loc[
            invalid_temperature,
            "product_name_uk",
        ].tolist()
        raise ValueError(
            "Цільова температура поза робочим діапазоном для: "
            + ", ".join(map(str, bad))
        )

    return result.sort_values("product_name_uk").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_products(sheet_url: str) -> pd.DataFrame:
    """Завантажує довідник продуктів із публічної Google-таблиці."""
    csv_url = google_sheet_to_csv_url(sheet_url)

    try:
        response = requests.get(csv_url, timeout=45)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Не вдалося завантажити Google-таблицю. Перевірте, що аркуш "
            "опублікований у форматі CSV та доступний без авторизації."
        ) from exc

    try:
        df = pd.read_csv(BytesIO(response.content))
    except Exception as exc:
        raise RuntimeError(
            "Отриману відповідь не вдалося прочитати як CSV."
        ) from exc

    return validate_product_table(df)


def calculate_mass_balance(
    mass_kg: float,
    initial_moisture_pct: float,
    final_moisture_pct: float,
) -> dict[str, float]:
    """Масовий баланс для вологості на вологій основі."""
    w0 = initial_moisture_pct / 100.0
    wf = final_moisture_pct / 100.0

    dry_matter = mass_kg * (1.0 - w0)
    initial_water = mass_kg * w0
    final_mass = dry_matter / (1.0 - wf)
    final_water = final_mass * wf
    water_to_remove = initial_water - final_water

    return {
        "dry_matter_kg": dry_matter,
        "initial_water_kg": initial_water,
        "final_mass_kg": final_mass,
        "final_water_kg": final_water,
        "water_to_remove_kg": water_to_remove,
        "initial_dry_basis": w0 / (1.0 - w0),
        "final_dry_basis": wf / (1.0 - wf),
    }


def infer_timezone(latitude: float, longitude: float) -> str:
    """Визначає часовий пояс за координатами."""
    result = TimezoneFinder().timezone_at(
        lat=float(latitude),
        lng=float(longitude),
    )
    return result or "UTC"


def parse_local_datetime(
    text: str,
    timezone_name: str,
) -> datetime:
    """Розбирає дату і час формату РРРР-ММ-ДД ГГ:ХХ."""
    try:
        naive = datetime.strptime(text.strip(), "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError(
            "Дата і час мають бути у форматі РРРР-ММ-ДД ГГ:ХХ."
        ) from exc

    return naive.replace(tzinfo=ZoneInfo(timezone_name))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasa_power(
    latitude: float,
    longitude: float,
    start_text: str,
    end_text: str,
) -> tuple[pd.DataFrame, str, str]:
    """Завантажує погодинні дані NASA POWER напряму через API."""
    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Широта повинна бути в межах від -90 до 90°.")
    if not -180 <= longitude <= 180:
        raise ValueError("Довгота повинна бути в межах від -180 до 180°.")

    timezone_name = infer_timezone(latitude, longitude)
    start_local = parse_local_datetime(start_text, timezone_name)
    end_local = parse_local_datetime(end_text, timezone_name)

    if end_local <= start_local:
        raise ValueError(
            "Час завершення повинен бути пізнішим за час початку."
        )

    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    end_utc = end_local.astimezone(ZoneInfo("UTC"))

    query = {
        "parameters": ",".join(NASA_PARAMETERS.keys()),
        "community": "RE",
        "latitude": latitude,
        "longitude": longitude,
        "start": start_utc.strftime("%Y%m%d"),
        "end": end_utc.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }

    try:
        response = requests.get(
            NASA_POWER_URL,
            params=query,
            timeout=90,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Не вдалося отримати дані NASA POWER: {exc}"
        ) from exc

    try:
        payload: dict[str, Any] = response.json()
        parameter_data = payload["properties"]["parameter"]
    except (ValueError, KeyError) as exc:
        raise RuntimeError(
            "NASA POWER не повернув очікуваний блок даних."
        ) from exc

    frame = pd.DataFrame(parameter_data)
    frame.index = pd.to_datetime(
        frame.index,
        format="%Y%m%d%H",
        utc=True,
    )
    frame.index = frame.index.tz_convert(timezone_name)
    frame.index.name = "time_local"

    frame = frame.replace(-999, pd.NA)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()

    frame = frame.loc[
        (frame.index >= start_local)
        & (frame.index <= end_local)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "Після приведення до місцевого часу дані відсутні."
        )

    status = (
        f"Отримано {len(frame)} погодинних записів. "
        f"Часовий пояс: {timezone_name}."
    )

    return frame, status, response.url


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Готує CSV у кодуванні UTF-8 з BOM для Excel."""
    return df.reset_index().to_csv(
        index=False
    ).encode("utf-8-sig")


st.set_page_config(
    page_title="Модель сонячної сушарки",
    page_icon="☀️",
    layout="wide",
)

st.title("Початкові дані моделі сонячної сушарки")
st.caption(
    "Довідник продуктів — Google Sheets. "
    "Метеорологічні дані — NASA POWER."
)

try:
    sheet_url = get_products_sheet_url()
    products = load_products(sheet_url)
except Exception as exc:
    st.error(str(exc))
    st.stop()

tab_product, tab_weather = st.tabs(
    ["1. Продукт", "2. NASA POWER"]
)

with tab_product:
    col1, col2 = st.columns(2)

    with col1:
        selected_name = st.selectbox(
            "Продукт",
            products["product_name_uk"].tolist(),
        )

    with col2:
        initial_mass_kg = st.number_input(
            "Початкова маса продукту, кг",
            min_value=0.001,
            value=30.0,
            step=1.0,
            format="%.3f",
        )

    product = products.loc[
        products["product_name_uk"] == selected_name
    ].iloc[0]

    if str(product["data_status"]).strip().lower() not in {
        "перевірено",
        "verified",
    }:
        st.warning(
            "Параметри продукту позначені як попередні. "
            "Перед використанням у дисертації їх потрібно підтвердити."
        )

    product_table = pd.DataFrame(
        {
            "Параметр": [
                "Форма продукту",
                "Основа вологості",
                "Початкова вологість, %",
                "Кінцева вологість, %",
                "Цільова температура сушіння, °C",
                "Робочий діапазон температур, °C",
                "Максимальна температура продукту, °C",
                "Нічний режим",
                "Запас до точки роси, °C",
                "Статус даних",
                "Джерело",
                "Посилання",
                "Примітка",
            ],
            "Значення": [
                product["product_form"],
                product["moisture_basis"],
                product["initial_moisture_pct"],
                product["final_moisture_pct"],
                product["target_drying_temp_c"],
                (
                    f"{product['working_temp_min_c']}–"
                    f"{product['working_temp_max_c']}"
                ),
                product["max_product_temp_c"],
                product["night_control_mode"],
                product["dew_point_margin_c"],
                product["data_status"],
                product["source_title"],
                product["source_url"],
                product["notes"],
            ],
        }
    )

    st.dataframe(
        product_table,
        use_container_width=True,
        hide_index=True,
    )

    balance = calculate_mass_balance(
        initial_mass_kg,
        float(product["initial_moisture_pct"]),
        float(product["final_moisture_pct"]),
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Маса сухої речовини",
        f"{balance['dry_matter_kg']:.3f} кг",
    )
    m2.metric(
        "Маса води до видалення",
        f"{balance['water_to_remove_kg']:.3f} кг",
    )
    m3.metric(
        "Кінцева маса продукту",
        f"{balance['final_mass_kg']:.3f} кг",
    )

    balance_table = pd.DataFrame(
        {
            "Показник": [
                "Початкова маса продукту",
                "Маса сухої речовини",
                "Початкова маса води",
                "Кінцева маса води",
                "Розрахункова кінцева маса продукту",
                "Маса води, яку необхідно видалити",
                "Початковий вологовміст на сухій основі",
                "Кінцевий вологовміст на сухій основі",
            ],
            "Значення": [
                initial_mass_kg,
                balance["dry_matter_kg"],
                balance["initial_water_kg"],
                balance["final_water_kg"],
                balance["final_mass_kg"],
                balance["water_to_remove_kg"],
                balance["initial_dry_basis"],
                balance["final_dry_basis"],
            ],
            "Одиниця": [
                "кг",
                "кг",
                "кг",
                "кг",
                "кг",
                "кг",
                "кг води/кг сухої речовини",
                "кг води/кг сухої речовини",
            ],
        }
    )

    st.dataframe(
        balance_table,
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Оновити довідник із Google Sheets"):
        st.cache_data.clear()
        st.rerun()

with tab_weather:
    c1, c2 = st.columns(2)

    with c1:
        latitude = st.number_input(
            "Широта, °",
            min_value=-90.0,
            max_value=90.0,
            value=50.446236,
            step=0.000001,
            format="%.6f",
        )

    with c2:
        longitude = st.number_input(
            "Довгота, °",
            min_value=-180.0,
            max_value=180.0,
            value=30.460662,
            step=0.000001,
            format="%.6f",
        )

    c3, c4 = st.columns(2)

    with c3:
        start_text = st.text_input(
            "Початок, місцевий час",
            value="2025-09-01 08:00",
        )

    with c4:
        end_text = st.text_input(
            "Завершення, місцевий час",
            value="2025-09-02 20:00",
        )

    st.dataframe(
        pd.DataFrame(
            {
                "Код NASA POWER": list(NASA_PARAMETERS.keys()),
                "Параметр": list(NASA_PARAMETERS.values()),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Завантажити дані NASA POWER",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Отримання даних NASA POWER..."):
            try:
                nasa_df, status, request_url = fetch_nasa_power(
                    latitude,
                    longitude,
                    start_text,
                    end_text,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["nasa_df"] = nasa_df
                st.session_state["nasa_status"] = status
                st.session_state["nasa_request_url"] = request_url

    if "nasa_df" in st.session_state:
        nasa_df = st.session_state["nasa_df"]

        st.success(st.session_state["nasa_status"])
        st.dataframe(
            nasa_df,
            use_container_width=True,
        )

        with st.expander("Адреса запиту NASA POWER"):
            st.code(st.session_state["nasa_request_url"])

        st.download_button(
            "Завантажити CSV",
            data=dataframe_to_csv_bytes(nasa_df),
            file_name="nasa_power_hourly.csv",
            mime="text/csv",
            use_container_width=True,
        )
