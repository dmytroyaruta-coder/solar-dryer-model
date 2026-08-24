from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timedelta
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

SOLAR_COLUMNS = [
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
]

METEO_COLUMNS = [
    "T2M",
    "RH2M",
    "T2MDEW",
    "WS10M",
    "PS",
]

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


def combine_local_datetime(
    selected_date: date,
    selected_time: time,
    timezone_name: str,
) -> datetime:
    """Формує локальний datetime із заданим часовим поясом."""
    return datetime.combine(
        selected_date,
        selected_time,
        tzinfo=ZoneInfo(timezone_name),
    )


def is_day_mode(
    timestamp: pd.Timestamp,
    day_start: time,
    day_end: time,
) -> bool:
    """
    Визначає режим роботи за локальним часом.
    Підтримує також інтервал, що перетинає опівніч.
    """
    current = timestamp.timetz().replace(tzinfo=None)

    if day_start < day_end:
        return day_start <= current < day_end

    return current >= day_start or current < day_end


def build_process_timeline(
    start_local: datetime,
    duration_hours: float,
    step_minutes: int,
    day_start: time,
    day_end: time,
) -> pd.DataFrame:
    """
    Формує часову шкалу моделі.
    Кожний рядок відповідає початку розрахункового інтервалу.
    """
    duration_minutes = round(duration_hours * 60)

    if duration_minutes <= 0:
        raise ValueError("Тривалість процесу повинна бути більшою за нуль.")

    if duration_minutes % step_minutes != 0:
        raise ValueError(
            "Тривалість процесу повинна ділитися на часовий крок без остачі."
        )

    intervals = duration_minutes // step_minutes
    end_local = start_local + timedelta(minutes=duration_minutes)

    index = pd.date_range(
        start=start_local,
        periods=intervals,
        freq=f"{step_minutes}min",
    )

    timeline = pd.DataFrame(index=index)
    timeline.index.name = "time_local"

    timeline["operating_mode"] = [
        "Денний режим"
        if is_day_mode(ts, day_start, day_end)
        else "Нічний режим"
        for ts in timeline.index
    ]

    timeline["interval_minutes"] = step_minutes
    timeline["elapsed_hours"] = (
        (timeline.index - timeline.index[0])
        .total_seconds()
        / 3600
    )

    timeline.attrs["start_local"] = start_local
    timeline.attrs["end_local"] = end_local
    timeline.attrs["duration_hours"] = duration_hours
    timeline.attrs["step_minutes"] = step_minutes
    timeline.attrs["intervals"] = intervals

    return timeline


def summarize_process_timeline(
    timeline: pd.DataFrame,
    step_minutes: int,
) -> dict[str, float]:
    """Обчислює тривалість денного та нічного режимів."""
    day_intervals = int(
        (timeline["operating_mode"] == "Денний режим").sum()
    )
    night_intervals = int(
        (timeline["operating_mode"] == "Нічний режим").sum()
    )

    return {
        "intervals": len(timeline),
        "day_intervals": day_intervals,
        "night_intervals": night_intervals,
        "day_hours": day_intervals * step_minutes / 60,
        "night_hours": night_intervals * step_minutes / 60,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasa_power(
    latitude: float,
    longitude: float,
    start_local_iso: str,
    end_local_iso: str,
    timezone_name: str,
) -> tuple[pd.DataFrame, str, str, dict[str, Any]]:
    """Завантажує погодинні дані NASA POWER напряму через API."""
    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Широта повинна бути в межах від -90 до 90°.")
    if not -180 <= longitude <= 180:
        raise ValueError("Довгота повинна бути в межах від -180 до 180°.")

    start_local = datetime.fromisoformat(start_local_iso)
    end_local = datetime.fromisoformat(end_local_iso)

    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    end_utc = end_local.astimezone(ZoneInfo("UTC"))

    # NASA POWER приймає календарні дати, тому завантажуємо повні доби,
    # які перекривають потрібний локальний інтервал.
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

    # Залишаємо невеликий запас довкола потрібного інтервалу.
    # Він потрібний для інтерполяції метеорологічних параметрів.
    buffer_start = start_local - timedelta(hours=1)
    buffer_end = end_local + timedelta(hours=1)

    frame = frame.loc[
        (frame.index >= buffer_start)
        & (frame.index <= buffer_end)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "Після приведення до місцевого часу дані відсутні."
        )

    status = (
        f"Отримано {len(frame)} погодинних записів NASA POWER. "
        f"Часовий пояс: {timezone_name}."
    )

    parameter_metadata = payload.get("parameters", {})

    return frame, status, response.url, parameter_metadata


def build_10min_weather(
    nasa_hourly: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Приводить погодинні дані NASA POWER до часової шкали моделі.

    Сонячне випромінювання:
    погодинне середнє значення поширюється на 10-хвилинні інтервали
    відповідної години (forward fill). Це зберігає погодинне середнє
    та не створює штучних внутрішньогодинних піків.

    Температура, вологість, точка роси, вітер і тиск:
    використовується лінійна інтерполяція в часі.
    """
    target_index = timeline.index
    combined_index = nasa_hourly.index.union(target_index).sort_values()

    result = pd.DataFrame(index=target_index)
    result.index.name = "time_local"

    solar = (
        nasa_hourly[SOLAR_COLUMNS]
        .reindex(combined_index)
        .ffill()
        .reindex(target_index)
    )

    meteo = (
        nasa_hourly[METEO_COLUMNS]
        .reindex(combined_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(target_index)
    )

    result = result.join(solar)
    result = result.join(meteo)
    result.insert(
        0,
        "operating_mode",
        timeline["operating_mode"],
    )
    result.insert(
        1,
        "elapsed_hours",
        timeline["elapsed_hours"],
    )

    return result



def saturation_vapor_pressure_kpa(temperature_c):
    """
    Тиск насиченої водяної пари над рідкою водою, кПа.

    Використано інженерне наближення типу Magnus для температур,
    характерних для сушіння та зовнішнього повітря.
    """
    import numpy as np

    t = pd.Series(temperature_c, dtype="float64")
    return 0.61094 * np.exp((17.625 * t) / (t + 243.04))


def calculate_psychrometric_state(
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Розраховує стан зовнішнього вологого повітря для кожного
    часового інтервалу моделі.

    Вихідні величини:
    - p_ws_kpa: тиск насиченої водяної пари, кПа;
    - p_v_kpa: парціальний тиск водяної пари, кПа;
    - humidity_ratio_kg_kg: вологовміст, кг води/кг сухого повітря;
    - enthalpy_kj_kg_da: ентальпія, кДж/кг сухого повітря;
    - dew_point_calc_c: розрахункова температура точки роси, °C;
    - dew_point_error_c: різниця між розрахунковою точкою роси
      та T2MDEW NASA POWER, °C;
    - moist_air_density_kg_m3: густина вологого повітря, кг/м³;
    - specific_volume_m3_kg_da: питомий об'єм, м³/кг сухого повітря.
    """
    import numpy as np

    required = {"T2M", "RH2M", "PS"}
    missing = required - set(weather_df.columns)
    if missing:
        raise ValueError(
            "Для психрометричного розрахунку відсутні стовпці: "
            + ", ".join(sorted(missing))
        )

    result = weather_df.copy()

    t_c = pd.to_numeric(result["T2M"], errors="coerce")
    rh_pct = pd.to_numeric(result["RH2M"], errors="coerce")
    p_kpa = pd.to_numeric(result["PS"], errors="coerce")

    if t_c.isna().any() or rh_pct.isna().any() or p_kpa.isna().any():
        raise ValueError(
            "У часовому ряді є пропущені T2M, RH2M або PS. "
            "Спочатку перевірте погодні дані."
        )

    if ((rh_pct < 0) | (rh_pct > 100)).any():
        raise ValueError(
            "Відносна вологість повинна бути в межах 0–100 %."
        )

    p_ws = saturation_vapor_pressure_kpa(t_c)
    p_v = (rh_pct / 100.0) * p_ws

    if (p_v >= p_kpa).any():
        raise ValueError(
            "Парціальний тиск водяної пари дорівнює або перевищує "
            "атмосферний тиск. Перевірте вихідні дані."
        )

    # Вологовміст, кг води / кг сухого повітря
    humidity_ratio = 0.621945 * p_v / (p_kpa - p_v)

    # Ентальпія вологого повітря, кДж / кг сухого повітря
    enthalpy = (
        1.006 * t_c
        + humidity_ratio * (2501.0 + 1.86 * t_c)
    )

    # Розрахункова температура точки роси через обернену формулу Magnus
    # Використовуємо RH не нижче 1e-6, щоб уникнути log(0).
    rh_fraction = (rh_pct / 100.0).clip(lower=1e-6)
    gamma = np.log(rh_fraction) + (17.625 * t_c) / (243.04 + t_c)
    dew_point_calc = 243.04 * gamma / (17.625 - gamma)

    # Густина вологого повітря через суму парціальних густин.
    # p у Па; температури в К.
    t_k = t_c + 273.15
    p_v_pa = p_v * 1000.0
    p_da_pa = (p_kpa - p_v) * 1000.0

    r_da = 287.055
    r_v = 461.495

    moist_air_density = (
        p_da_pa / (r_da * t_k)
        + p_v_pa / (r_v * t_k)
    )

    # Питомий об'єм на 1 кг сухого повітря
    specific_volume = (
        r_da * t_k * (1.0 + 1.607858 * humidity_ratio)
        / (p_kpa * 1000.0)
    )

    result["p_ws_kpa"] = p_ws
    result["p_v_kpa"] = p_v
    result["humidity_ratio_kg_kg"] = humidity_ratio
    result["humidity_ratio_g_kg"] = humidity_ratio * 1000.0
    result["enthalpy_kj_kg_da"] = enthalpy
    result["dew_point_calc_c"] = dew_point_calc
    result["moist_air_density_kg_m3"] = moist_air_density
    result["specific_volume_m3_kg_da"] = specific_volume

    if "T2MDEW" in result.columns:
        nasa_dew = pd.to_numeric(
            result["T2MDEW"],
            errors="coerce",
        )
        result["dew_point_error_c"] = (
            result["dew_point_calc_c"] - nasa_dew
        )
        result["dew_point_abs_error_c"] = (
            result["dew_point_error_c"].abs()
        )

    return result


def summarize_psychrometrics(
    psychro_df: pd.DataFrame,
) -> pd.DataFrame:
    """Формує коротку таблицю основних психрометричних показників."""
    rows = [
        (
            "Середній вологовміст",
            psychro_df["humidity_ratio_g_kg"].mean(),
            "г води/кг сухого повітря",
        ),
        (
            "Мінімальний вологовміст",
            psychro_df["humidity_ratio_g_kg"].min(),
            "г води/кг сухого повітря",
        ),
        (
            "Максимальний вологовміст",
            psychro_df["humidity_ratio_g_kg"].max(),
            "г води/кг сухого повітря",
        ),
        (
            "Середня ентальпія",
            psychro_df["enthalpy_kj_kg_da"].mean(),
            "кДж/кг сухого повітря",
        ),
        (
            "Мінімальна розрахункова точка роси",
            psychro_df["dew_point_calc_c"].min(),
            "°C",
        ),
        (
            "Максимальна розрахункова точка роси",
            psychro_df["dew_point_calc_c"].max(),
            "°C",
        ),
        (
            "Середня густина вологого повітря",
            psychro_df["moist_air_density_kg_m3"].mean(),
            "кг/м³",
        ),
    ]

    if "dew_point_abs_error_c" in psychro_df.columns:
        rows.append(
            (
                "Середнє абсолютне відхилення точки роси від NASA",
                psychro_df["dew_point_abs_error_c"].mean(),
                "°C",
            )
        )
        rows.append(
            (
                "Максимальне абсолютне відхилення точки роси від NASA",
                psychro_df["dew_point_abs_error_c"].max(),
                "°C",
            )
        )

    return pd.DataFrame(
        rows,
        columns=["Показник", "Значення", "Одиниця"],
    )


def calculate_required_moisture_removal_profile(
    timeline: pd.DataFrame,
    water_to_remove_kg: float,
    dry_matter_kg: float,
    initial_water_kg: float,
    step_minutes: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Пункт 10. Цільова середня швидкість видалення води.

    У денному режимі приймається постійна середня цільова швидкість,
    у нічному режимі — нульова. Це не фактична кінетика сушіння.
    """
    if water_to_remove_kg <= 0:
        raise ValueError("Маса води до видалення повинна бути > 0.")
    if dry_matter_kg <= 0:
        raise ValueError("Маса сухої речовини повинна бути > 0.")

    result = timeline.copy()
    active_mask = result["operating_mode"] == "Денний режим"
    active_intervals = int(active_mask.sum())

    if active_intervals == 0:
        raise ValueError("У циклі немає денних інтервалів сушіння.")

    active_hours = active_intervals * step_minutes / 60.0
    avg_rate_kg_h = water_to_remove_kg / active_hours
    water_per_interval_kg = avg_rate_kg_h * step_minutes / 60.0

    result["required_moisture_removal_rate_kg_h"] = 0.0
    result.loc[
        active_mask, "required_moisture_removal_rate_kg_h"
    ] = avg_rate_kg_h

    result["target_water_removed_interval_kg"] = 0.0
    result.loc[
        active_mask, "target_water_removed_interval_kg"
    ] = water_per_interval_kg

    result["target_cumulative_water_removed_kg"] = (
        result["target_water_removed_interval_kg"]
        .cumsum()
        .clip(upper=water_to_remove_kg)
    )

    final_water_kg = initial_water_kg - water_to_remove_kg

    result["target_water_remaining_kg"] = (
        initial_water_kg
        - result["target_cumulative_water_removed_kg"]
    ).clip(lower=final_water_kg)

    result["target_product_mass_kg"] = (
        dry_matter_kg + result["target_water_remaining_kg"]
    )

    result["target_product_moisture_wb_pct"] = (
        result["target_water_remaining_kg"]
        / result["target_product_mass_kg"]
        * 100.0
    )

    result["target_product_moisture_db_kg_kg"] = (
        result["target_water_remaining_kg"] / dry_matter_kg
    )

    summary = {
        "active_intervals": active_intervals,
        "active_hours": active_hours,
        "average_required_rate_kg_h": avg_rate_kg_h,
        "average_required_rate_g_h": avg_rate_kg_h * 1000.0,
        "average_required_rate_g_min": avg_rate_kg_h * 1000.0 / 60.0,
        "water_per_active_interval_kg": water_per_interval_kg,
        "water_per_active_interval_g": water_per_interval_kg * 1000.0,
    }

    return result, summary


def summarize_required_moisture_removal(
    water_to_remove_kg: float,
    step_minutes: int,
    summary: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Показник": [
                "Маса води, яку необхідно видалити",
                "Сумарна тривалість активного сушіння",
                "Середня потрібна швидкість видалення води",
                "Середня потрібна швидкість видалення води",
                "Середня потрібна швидкість видалення води",
                f"Вода за один активний інтервал ({step_minutes} хв)",
            ],
            "Значення": [
                water_to_remove_kg,
                summary["active_hours"],
                summary["average_required_rate_kg_h"],
                summary["average_required_rate_g_h"],
                summary["average_required_rate_g_min"],
                summary["water_per_active_interval_g"],
            ],
            "Одиниця": [
                "кг",
                "год",
                "кг/год",
                "г/год",
                "г/хв",
                "г/інтервал",
            ],
        }
    )


def calculate_dryer_geometry(
    product_mass_kg: float,
    chamber_length_m: float,
    chamber_width_m: float,
    chamber_height_m: float,
    tray_count: int,
    tray_length_m: float,
    tray_width_m: float,
    layer_thickness_m: float,
    tray_pitch_m: float,
    airflow_direction: str,
    reference_bulk_density_kg_m3: float | None = None,
) -> dict[str, float | str | None]:
    """
    Пункт 11. Геометричні параметри сушильної камери та шару продукту.

    Розрахунок не задає витрату повітря і не використовує припущення
    щодо стану повітря на виході. Формуються лише геометричні та
    масо-геометричні характеристики, потрібні для наступних етапів.
    """
    positive_values = {
        "Початкова маса продукту": product_mass_kg,
        "Довжина камери": chamber_length_m,
        "Ширина камери": chamber_width_m,
        "Висота камери": chamber_height_m,
        "Довжина лотка": tray_length_m,
        "Ширина лотка": tray_width_m,
        "Товщина шару": layer_thickness_m,
        "Крок між ярусами": tray_pitch_m,
    }

    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} має бути більше нуля.")

    if tray_count < 1:
        raise ValueError("Кількість лотків має бути не меншою за 1.")

    chamber_volume_m3 = (
        chamber_length_m
        * chamber_width_m
        * chamber_height_m
    )

    tray_area_m2 = tray_length_m * tray_width_m
    total_drying_area_m2 = tray_count * tray_area_m2

    product_layer_volume_m3 = (
        total_drying_area_m2
        * layer_thickness_m
    )

    specific_loading_kg_m2 = (
        product_mass_kg / total_drying_area_m2
    )

    calculated_bulk_loading_kg_m3 = (
        product_mass_kg / product_layer_volume_m3
    )

    chamber_fill_fraction_pct = (
        product_layer_volume_m3
        / chamber_volume_m3
        * 100.0
    )

    if airflow_direction == "Уздовж довжини камери":
        gross_flow_area_m2 = (
            chamber_width_m * chamber_height_m
        )
        flow_path_length_m = chamber_length_m

    elif airflow_direction == "Уздовж ширини камери":
        gross_flow_area_m2 = (
            chamber_length_m * chamber_height_m
        )
        flow_path_length_m = chamber_width_m

    elif airflow_direction == "Вертикально":
        gross_flow_area_m2 = (
            chamber_length_m * chamber_width_m
        )
        flow_path_length_m = chamber_height_m

    else:
        raise ValueError("Невідомий напрям основного потоку повітря.")

    # Геометрична перевірка розміщення лотка у плані.
    tray_fits_in_plan = (
        tray_length_m <= chamber_length_m
        and tray_width_m <= chamber_width_m
    )

    # Спрощена перевірка вертикального розміщення ярусів.
    # Товщина самих лотків і крайові зазори тут ще не враховуються.
    required_stack_height_m = (
        (tray_count - 1) * tray_pitch_m
        + layer_thickness_m
    )
    stack_fits_height = (
        required_stack_height_m <= chamber_height_m
    )

    required_layer_thickness_from_reference_m = None
    bulk_density_difference_pct = None

    if (
        reference_bulk_density_kg_m3 is not None
        and reference_bulk_density_kg_m3 > 0
    ):
        required_layer_thickness_from_reference_m = (
            product_mass_kg
            / (
                reference_bulk_density_kg_m3
                * total_drying_area_m2
            )
        )

        bulk_density_difference_pct = (
            (
                calculated_bulk_loading_kg_m3
                - reference_bulk_density_kg_m3
            )
            / reference_bulk_density_kg_m3
            * 100.0
        )

    return {
        "chamber_volume_m3": chamber_volume_m3,
        "tray_area_m2": tray_area_m2,
        "total_drying_area_m2": total_drying_area_m2,
        "product_layer_volume_m3": product_layer_volume_m3,
        "specific_loading_kg_m2": specific_loading_kg_m2,
        "calculated_bulk_loading_kg_m3": (
            calculated_bulk_loading_kg_m3
        ),
        "chamber_fill_fraction_pct": chamber_fill_fraction_pct,
        "gross_flow_area_m2": gross_flow_area_m2,
        "flow_path_length_m": flow_path_length_m,
        "required_stack_height_m": required_stack_height_m,
        "tray_fits_in_plan": tray_fits_in_plan,
        "stack_fits_height": stack_fits_height,
        "reference_bulk_density_kg_m3": (
            reference_bulk_density_kg_m3
        ),
        "required_layer_thickness_from_reference_m": (
            required_layer_thickness_from_reference_m
        ),
        "bulk_density_difference_pct": (
            bulk_density_difference_pct
        ),
    }


def relative_humidity_from_humidity_ratio(
    temperature_c,
    humidity_ratio_kg_kg,
    pressure_kpa,
):
    """
    Визначає відносну вологість повітря за температурою,
    вологовмістом і повним тиском.

    humidity_ratio_kg_kg: кг води / кг сухого повітря
    pressure_kpa: кПа
    """
    import numpy as np

    t = pd.Series(temperature_c, dtype="float64")
    w = pd.Series(humidity_ratio_kg_kg, dtype="float64")
    p = pd.Series(pressure_kpa, dtype="float64")

    p_v = w * p / (0.621945 + w)
    p_ws = saturation_vapor_pressure_kpa(t)

    rh = 100.0 * p_v / p_ws
    return rh.clip(lower=0.0, upper=100.0)


def modified_oswin_wheat_pionier(
    temperature_c,
    relative_humidity_pct,
) -> pd.DataFrame:
    """
    Рівноважний вологовміст пшениці Triticum aestivum L.,
    cv. 'Pionier' за Modified Oswin model, Ramaj et al. (2021).

    X_eq = (C1 + C2*T) *
           [(RH/100)/(1 - RH/100)]^(1/C3)

    Коефіцієнти:
        C1 = 0.129
        C2 = -6.460e-4 1/°C
        C3 = 2.944

    Діапазон експериментальної валідації:
        10 <= T <= 50 °C
        5.7 <= RH <= 86.8 %

    X_eq: кг води / кг сухої речовини (dry basis).
    """
    import numpy as np

    c1 = 0.129
    c2 = -6.460e-4
    c3 = 2.944

    t = pd.Series(temperature_c, dtype="float64")
    rh = pd.Series(relative_humidity_pct, dtype="float64")

    valid = (
        t.between(10.0, 50.0, inclusive="both")
        & rh.between(5.7, 86.8, inclusive="both")
    )

    aw = rh / 100.0

    x_eq = pd.Series(np.nan, index=t.index, dtype="float64")

    safe = valid & (aw > 0.0) & (aw < 1.0)

    x_eq.loc[safe] = (
        (c1 + c2 * t.loc[safe])
        * (
            aw.loc[safe]
            / (1.0 - aw.loc[safe])
        ) ** (1.0 / c3)
    )

    w_eq_wb_pct = (
        x_eq / (1.0 + x_eq) * 100.0
    )

    return pd.DataFrame(
        {
            "equilibrium_moisture_db_kg_kg": x_eq,
            "equilibrium_moisture_wb_pct": w_eq_wb_pct,
            "oswin_validity_ok": valid,
        },
        index=t.index,
    )


def calculate_equilibrium_moisture_profile(
    psychro_df: pd.DataFrame,
    drying_temperature_c: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Етап 13.

    Для активного денного режиму:
    1. Береться вологовміст зовнішнього повітря з етапу 9.
    2. Повітря умовно нагрівається до заданої температури сушіння
       без додавання/видалення вологи, тому d = const.
    3. За d, p і T_суш визначається RH сушильного агента на вході.
    4. За Modified Oswin визначається X_eq пшениці.

    У нічному режимі X_eq тут не розраховується, оскільки нічний
    температурний режим ще не сформований як режим активного сушіння.
    """
    import numpy as np

    required = {
        "humidity_ratio_kg_kg",
        "PS",
        "operating_mode",
    }
    missing = required - set(psychro_df.columns)

    if missing:
        raise ValueError(
            "Для етапу 13 відсутні параметри з етапу 9: "
            + ", ".join(sorted(missing))
        )

    result = psychro_df.copy()

    day_mask = result["operating_mode"] == "Денний режим"

    result["drying_air_temperature_c"] = np.nan
    result.loc[
        day_mask,
        "drying_air_temperature_c",
    ] = float(drying_temperature_c)

    rh_drying = pd.Series(
        np.nan,
        index=result.index,
        dtype="float64",
    )

    if day_mask.any():
        rh_calc = relative_humidity_from_humidity_ratio(
            temperature_c=pd.Series(
                float(drying_temperature_c),
                index=result.index[day_mask],
            ),
            humidity_ratio_kg_kg=result.loc[
                day_mask,
                "humidity_ratio_kg_kg",
            ],
            pressure_kpa=result.loc[
                day_mask,
                "PS",
            ],
        )
        rh_calc.index = result.index[day_mask]
        rh_drying.loc[day_mask] = rh_calc

    result["drying_air_inlet_rh_pct"] = rh_drying

    oswin = modified_oswin_wheat_pionier(
        temperature_c=result["drying_air_temperature_c"],
        relative_humidity_pct=result["drying_air_inlet_rh_pct"],
    )

    result["equilibrium_moisture_db_kg_kg"] = (
        oswin["equilibrium_moisture_db_kg_kg"]
    )
    result["equilibrium_moisture_wb_pct"] = (
        oswin["equilibrium_moisture_wb_pct"]
    )
    result["oswin_validity_ok"] = (
        oswin["oswin_validity_ok"]
    )

    active = result.loc[day_mask].copy()
    valid_active = active.loc[
        active["oswin_validity_ok"]
    ].copy()

    invalid_active_count = int(
        (~active["oswin_validity_ok"]).sum()
    )

    if valid_active.empty:
        summary = {
            "mean_drying_air_rh_pct": float("nan"),
            "min_drying_air_rh_pct": float("nan"),
            "max_drying_air_rh_pct": float("nan"),
            "mean_xeq_db_kg_kg": float("nan"),
            "min_xeq_db_kg_kg": float("nan"),
            "max_xeq_db_kg_kg": float("nan"),
            "mean_weq_wb_pct": float("nan"),
            "invalid_active_intervals": invalid_active_count,
        }
    else:
        summary = {
            "mean_drying_air_rh_pct": float(
                valid_active["drying_air_inlet_rh_pct"].mean()
            ),
            "min_drying_air_rh_pct": float(
                valid_active["drying_air_inlet_rh_pct"].min()
            ),
            "max_drying_air_rh_pct": float(
                valid_active["drying_air_inlet_rh_pct"].max()
            ),
            "mean_xeq_db_kg_kg": float(
                valid_active[
                    "equilibrium_moisture_db_kg_kg"
                ].mean()
            ),
            "min_xeq_db_kg_kg": float(
                valid_active[
                    "equilibrium_moisture_db_kg_kg"
                ].min()
            ),
            "max_xeq_db_kg_kg": float(
                valid_active[
                    "equilibrium_moisture_db_kg_kg"
                ].max()
            ),
            "mean_weq_wb_pct": float(
                valid_active[
                    "equilibrium_moisture_wb_pct"
                ].mean()
            ),
            "invalid_active_intervals": invalid_active_count,
        }

    return result, summary


def page_k_wheat_pionier(
    temperature_c,
    relative_humidity_pct,
    air_velocity_m_s: float,
):
    """
    Generalized Page k parameter for wheat cv. 'Pionier'
    according to Ramaj et al. (2021):

        k = 2.8e-3 * exp(0.059*T) * RH^(-0.139) * v^(0.025)

    T  : °C
    RH : %
    v  : m/s

    Experimental Page domain reported by the source:
        T = 10...50 °C
        RH = 20...60 %
        v = 0.15...1.00 m/s

    Policy used here:
    - k is "validated" when all three variables are within the
      experimental Page domain;
    - if T and v are within the source range but RH is outside
      20...60 %, the same correlation is evaluated and explicitly
      marked as "extrapolated_rh";
    - if T or v are outside the source range, k is not calculated.

    This preserves the real calculated RH instead of artificially
    clipping it to 20 or 60 %.
    """
    import numpy as np

    t = pd.Series(temperature_c, dtype="float64")
    rh = pd.Series(relative_humidity_pct, dtype="float64")

    tv_valid = (
        t.between(10.0, 50.0, inclusive="both")
        & (0.15 <= float(air_velocity_m_s) <= 1.00)
    )

    rh_physical = (
        (rh > 0.0)
        & (rh < 100.0)
    )

    validated = (
        tv_valid
        & rh.between(20.0, 60.0, inclusive="both")
    )

    extrapolated_rh = (
        tv_valid
        & rh_physical
        & ~rh.between(20.0, 60.0, inclusive="both")
    )

    calculable = validated | extrapolated_rh

    k = pd.Series(
        np.nan,
        index=t.index,
        dtype="float64",
    )

    k.loc[calculable] = (
        2.8e-3
        * np.exp(0.059 * t.loc[calculable])
        * rh.loc[calculable] ** (-0.139)
        * float(air_velocity_m_s) ** 0.025
    )

    # Diagnostic comparison only:
    # what k would be if RH were limited to the nearest validated
    # Page boundary (20 or 60 %). This is NOT used as the model input.
    rh_clipped = rh.clip(lower=20.0, upper=60.0)

    k_rh_clipped = pd.Series(
        np.nan,
        index=t.index,
        dtype="float64",
    )

    clipped_calculable = tv_valid & rh_physical

    k_rh_clipped.loc[clipped_calculable] = (
        2.8e-3
        * np.exp(0.059 * t.loc[clipped_calculable])
        * rh_clipped.loc[clipped_calculable] ** (-0.139)
        * float(air_velocity_m_s) ** 0.025
    )

    k_difference_pct = (
        (k - k_rh_clipped)
        / k_rh_clipped
        * 100.0
    )

    status = pd.Series(
        "outside_model_domain",
        index=t.index,
        dtype="object",
    )
    status.loc[validated] = "validated"
    status.loc[extrapolated_rh] = "extrapolated_rh"

    return pd.DataFrame(
        {
            "page_k_min_inv": k,
            "page_n": 0.784,
            "page_status": status,
            "page_validity_ok": validated,
            "page_extrapolated_rh": extrapolated_rh,
            "page_rh_clipped_for_check_pct": rh_clipped,
            "page_k_rh_clipped_check_min_inv": k_rh_clipped,
            "page_k_extrapolation_difference_pct": k_difference_pct,
        },
        index=t.index,
    )

def prepare_heat_mass_transfer_inputs(
    equilibrium_profile: pd.DataFrame,
    air_velocity_m_s: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Stage 14: prepares time-dependent quantities for the coupled
    heat-and-mass-transfer system.

    Page:
    - validated: RH=20...60 %, T=10...50 °C, v=0.15...1.00 m/s;
    - extrapolated_rh: T and v are inside the source range, but RH
      is outside 20...60 %. k is calculated and clearly flagged;
    - outside_model_domain: T or v is outside the source range,
      or RH is nonphysical.

    Modified Oswin validity remains an independent check.
    """
    required = {
        "operating_mode",
        "drying_air_temperature_c",
        "drying_air_inlet_rh_pct",
        "equilibrium_moisture_db_kg_kg",
        "oswin_validity_ok",
    }

    missing = required - set(equilibrium_profile.columns)
    if missing:
        raise ValueError(
            "Для етапу 14 відсутні результати попередніх етапів: "
            + ", ".join(sorted(missing))
        )

    result = equilibrium_profile.copy()

    page = page_k_wheat_pionier(
        temperature_c=result["drying_air_temperature_c"],
        relative_humidity_pct=result["drying_air_inlet_rh_pct"],
        air_velocity_m_s=air_velocity_m_s,
    )

    for col in page.columns:
        result[col] = page[col]

    result["air_velocity_m_s"] = float(air_velocity_m_s)

    # Combined status for later use of Page + Modified Oswin.
    result["drying_model_status"] = "outside_model_domain"

    oswin_ok = result["oswin_validity_ok"].fillna(False)
    page_valid = result["page_status"] == "validated"
    page_extrap = result["page_status"] == "extrapolated_rh"

    result.loc[
        oswin_ok & page_valid,
        "drying_model_status",
    ] = "validated"

    result.loc[
        oswin_ok & page_extrap,
        "drying_model_status",
    ] = "page_rh_extrapolation"

    result.loc[
        ~oswin_ok,
        "drying_model_status",
    ] = "outside_oswin_domain"

    day_mask = result["operating_mode"] == "Денний режим"
    active = result.loc[day_mask]

    validated_active = active.loc[
        active["drying_model_status"] == "validated"
    ]

    extrapolated_active = active.loc[
        active["drying_model_status"] == "page_rh_extrapolation"
    ]

    usable_active = active.loc[
        active["drying_model_status"].isin(
            ["validated", "page_rh_extrapolation"]
        )
    ]

    unusable_active = active.loc[
        ~active["drying_model_status"].isin(
            ["validated", "page_rh_extrapolation"]
        )
    ]

    extrap_diff = pd.to_numeric(
        extrapolated_active[
            "page_k_extrapolation_difference_pct"
        ],
        errors="coerce",
    ).abs()

    summary = {
        "air_velocity_m_s": float(air_velocity_m_s),
        "active_intervals": int(day_mask.sum()),
        "validated_active_intervals": int(
            validated_active.shape[0]
        ),
        "extrapolated_active_intervals": int(
            extrapolated_active.shape[0]
        ),
        "usable_active_intervals": int(
            usable_active.shape[0]
        ),
        "unusable_active_intervals": int(
            unusable_active.shape[0]
        ),
        "mean_k_min_inv": (
            float(usable_active["page_k_min_inv"].mean())
            if not usable_active.empty
            else float("nan")
        ),
        "min_k_min_inv": (
            float(usable_active["page_k_min_inv"].min())
            if not usable_active.empty
            else float("nan")
        ),
        "max_k_min_inv": (
            float(usable_active["page_k_min_inv"].max())
            if not usable_active.empty
            else float("nan")
        ),
        "mean_abs_k_extrapolation_difference_pct": (
            float(extrap_diff.mean())
            if not extrap_diff.empty
            else 0.0
        ),
        "max_abs_k_extrapolation_difference_pct": (
            float(extrap_diff.max())
            if not extrap_diff.empty
            else 0.0
        ),
    }

    return result, summary

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

st.title("Модель комплексної сонячної сушарки")
st.caption(
    "Етапи 1–14: вихідні дані, масовий баланс, режими, "
    "погодні дані, психрометрія, геометрія, кінетика Page, "
    "рівноважна вологість та формування системи "
    "тепло- і масообміну сушильної камери."
)

try:
    sheet_url = get_products_sheet_url()
    products = load_products(sheet_url)
except Exception as exc:
    st.error(str(exc))
    st.stop()

tab_product, tab_process, tab_weather, tab_psychro, tab_removal, tab_geometry, tab_kinetics, tab_equilibrium, tab_heat_mass = st.tabs(
    [
        "1. Продукт",
        "2. Тривалість і режими",
        "3. Погодні дані NASA POWER",
        "4. Стан вологого повітря",
        "5. Потрібна швидкість сушіння",
        "6. Геометрія камери і шару",
        "7. Кінетична модель сушіння",
        "8. Рівноважна вологість",
        "9. Тепло- і масообмін",
    ]
)

# ---------------------------------------------------------------------
# 1. ПРОДУКТ
# ---------------------------------------------------------------------
with tab_product:
    st.subheader("Етапи 1–5. Вихідні дані продукту")

    col1, col2 = st.columns(2)

    with col1:
        selected_name = st.selectbox(
            "Продукт",
            products["product_name_uk"].tolist(),
            key="selected_product",
        )

    with col2:
        initial_mass_kg = st.number_input(
            "Початкова маса продукту, кг",
            min_value=0.001,
            value=30.0,
            step=1.0,
            format="%.3f",
            key="initial_mass_kg",
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

    st.session_state["mass_balance"] = balance

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

    if st.button(
        "Оновити довідник із Google Sheets",
        key="refresh_products",
    ):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------
# 2. ТРИВАЛІСТЬ І РЕЖИМИ
# ---------------------------------------------------------------------
with tab_process:
    st.subheader("Етапи 6–7. Тривалість процесу та денний/нічний режими")

    st.info(
        "На цьому етапі задається розрахунковий горизонт моделі. "
        "Це не означає, що цільова вологість обов'язково буде досягнута "
        "саме за цей час — надалі це перевірятиме модель сушіння."
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        process_start_date = st.date_input(
            "Дата початку процесу",
            value=date(2025, 9, 1),
            key="process_start_date",
        )

    with p2:
        process_start_time = st.time_input(
            "Час початку процесу",
            value=time(8, 0),
            step=600,
            key="process_start_time",
        )

    with p3:
        process_duration_hours = st.number_input(
            "Задана тривалість процесу, год",
            min_value=1.0,
            max_value=720.0,
            value=36.0,
            step=1.0,
            key="process_duration_hours",
        )

    r1, r2, r3 = st.columns(3)

    with r1:
        day_start_time = st.time_input(
            "Початок денного режиму",
            value=time(8, 0),
            step=600,
            key="day_start_time",
        )

    with r2:
        day_end_time = st.time_input(
            "Завершення денного режиму",
            value=time(20, 0),
            step=600,
            key="day_end_time",
        )

    with r3:
        model_step_minutes = st.selectbox(
            "Часовий крок моделі, хв",
            options=[5, 10, 15, 30, 60],
            index=1,
            key="model_step_minutes",
        )

    # Для попереднього відображення часовий пояс ще не потрібний.
    # Використовуємо UTC як технічний tz; у вкладці NASA часова шкала
    # буде перебудована в часовому поясі координат користувача.
    preview_start = datetime.combine(
        process_start_date,
        process_start_time,
        tzinfo=ZoneInfo("UTC"),
    )

    try:
        preview_timeline = build_process_timeline(
            start_local=preview_start,
            duration_hours=process_duration_hours,
            step_minutes=model_step_minutes,
            day_start=day_start_time,
            day_end=day_end_time,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    preview_summary = summarize_process_timeline(
        preview_timeline,
        model_step_minutes,
    )

    process_end_preview = (
        preview_start
        + timedelta(hours=float(process_duration_hours))
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Розрахункових інтервалів",
        f"{preview_summary['intervals']}",
    )
    s2.metric(
        "Активний денний режим",
        f"{preview_summary['day_hours']:.1f} год",
    )
    s3.metric(
        "Нічний режим",
        f"{preview_summary['night_hours']:.1f} год",
    )
    s4.metric(
        "Крок моделі",
        f"{model_step_minutes} хв",
    )

    process_summary_table = pd.DataFrame(
        {
            "Параметр": [
                "Початок процесу",
                "Завершення процесу",
                "Задана тривалість",
                "Початок денного режиму",
                "Завершення денного режиму",
                "Тривалість активного денного режиму",
                "Тривалість нічного режиму",
                "Кількість розрахункових інтервалів",
            ],
            "Значення": [
                f"{process_start_date} {process_start_time:%H:%M}",
                f"{process_end_preview.date()} {process_end_preview.time():%H:%M}",
                f"{process_duration_hours:.1f} год",
                f"{day_start_time:%H:%M}",
                f"{day_end_time:%H:%M}",
                f"{preview_summary['day_hours']:.1f} год",
                f"{preview_summary['night_hours']:.1f} год",
                preview_summary["intervals"],
            ],
        }
    )

    st.dataframe(
        process_summary_table,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Показати часову шкалу режимів"):
        preview_display = preview_timeline.reset_index().copy()
        preview_display["time_local"] = (
            preview_display["time_local"]
            .dt.tz_localize(None)
        )
        st.dataframe(
            preview_display,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------
# 3. NASA POWER ТА 10-ХВИЛИННИЙ РЯД
# ---------------------------------------------------------------------
with tab_weather:
    st.subheader("Етап 8. Погодні часові ряди")

    c1, c2 = st.columns(2)

    with c1:
        latitude = st.number_input(
            "Широта, °",
            min_value=-90.0,
            max_value=90.0,
            value=50.446236,
            step=0.000001,
            format="%.6f",
            key="latitude",
        )

    with c2:
        longitude = st.number_input(
            "Довгота, °",
            min_value=-180.0,
            max_value=180.0,
            value=30.460662,
            step=0.000001,
            format="%.6f",
            key="longitude",
        )

    timezone_name = infer_timezone(latitude, longitude)

    start_local = combine_local_datetime(
        process_start_date,
        process_start_time,
        timezone_name,
    )
    end_local = start_local + timedelta(
        hours=float(process_duration_hours)
    )

    st.write(
        f"Визначений часовий пояс: `{timezone_name}`. "
        f"Розрахунковий інтервал: "
        f"**{start_local:%d.%m.%Y %H:%M} – {end_local:%d.%m.%Y %H:%M}**."
    )

    try:
        model_timeline = build_process_timeline(
            start_local=start_local,
            duration_hours=process_duration_hours,
            step_minutes=model_step_minutes,
            day_start=day_start_time,
            day_end=day_end_time,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    model_summary = summarize_process_timeline(
        model_timeline,
        model_step_minutes,
    )

    n1, n2, n3 = st.columns(3)
    n1.metric(
        "Тривалість ряду",
        f"{process_duration_hours:.1f} год",
    )
    n2.metric(
        "Часовий крок моделі",
        f"{model_step_minutes} хв",
    )
    n3.metric(
        "Кількість інтервалів",
        f"{model_summary['intervals']}",
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

    st.caption(
        "NASA POWER надає погодинні середні значення. "
        "У моделі сонячні показники поширюються на 10-хвилинні "
        "інтервали відповідної години без створення штучних піків; "
        "температура, відносна вологість, точка роси, вітер і тиск "
        "лінійно інтерполюються між сусідніми погодинними значеннями."
    )

    if st.button(
        "Завантажити NASA POWER і сформувати ряд моделі",
        type="primary",
        use_container_width=True,
        key="load_nasa",
    ):
        with st.spinner("Отримання та опрацювання даних NASA POWER..."):
            try:
                nasa_hourly, status, request_url, parameter_metadata = (
                    fetch_nasa_power(
                        latitude=latitude,
                        longitude=longitude,
                        start_local_iso=start_local.isoformat(),
                        end_local_iso=end_local.isoformat(),
                        timezone_name=timezone_name,
                    )
                )

                weather_model = build_10min_weather(
                    nasa_hourly=nasa_hourly,
                    timeline=model_timeline,
                )

            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["nasa_hourly"] = nasa_hourly
                st.session_state["weather_model"] = weather_model
                st.session_state["nasa_status"] = status
                st.session_state["nasa_request_url"] = request_url
                st.session_state["nasa_parameter_metadata"] = (
                    parameter_metadata
                )
                st.session_state["weather_timezone"] = timezone_name

    if "weather_model" in st.session_state:
        nasa_hourly = st.session_state["nasa_hourly"]
        weather_model = st.session_state["weather_model"]

        st.success(
            st.session_state["nasa_status"]
            + f" Сформовано {len(weather_model)} "
            f"розрахункових інтервалів по {model_step_minutes} хв."
        )

        st.markdown("#### Погодинні вихідні дані NASA POWER")
        st.dataframe(
            nasa_hourly,
            use_container_width=True,
        )

        st.markdown("#### Часовий ряд математичної моделі")
        st.dataframe(
            weather_model,
            use_container_width=True,
        )

        st.markdown("#### Сонячне випромінювання")
        st.line_chart(
            weather_model[
                [
                    "ALLSKY_SFC_SW_DWN",
                    "ALLSKY_SFC_SW_DNI",
                    "ALLSKY_SFC_SW_DIFF",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("#### Температура зовнішнього повітря")
        st.line_chart(
            weather_model[["T2M", "T2MDEW"]],
            use_container_width=True,
        )

        st.markdown("#### Відносна вологість зовнішнього повітря")
        st.line_chart(
            weather_model[["RH2M"]],
            use_container_width=True,
        )

        with st.expander("Метадані та адреса запиту NASA POWER"):
            st.code(st.session_state["nasa_request_url"])
            metadata = st.session_state.get(
                "nasa_parameter_metadata",
                {},
            )
            if metadata:
                metadata_rows = []
                for code, meta in metadata.items():
                    if isinstance(meta, dict):
                        metadata_rows.append(
                            {
                                "Код": code,
                                "Назва": meta.get("longname", ""),
                                "Одиниця": meta.get("units", ""),
                            }
                        )
                if metadata_rows:
                    st.dataframe(
                        pd.DataFrame(metadata_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "Завантажити погодинні дані NASA POWER",
                data=dataframe_to_csv_bytes(nasa_hourly),
                file_name="nasa_power_hourly.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_nasa_hourly",
            )

        with d2:
            st.download_button(
                "Завантажити часовий ряд моделі",
                data=dataframe_to_csv_bytes(weather_model),
                file_name=(
                    f"weather_model_{model_step_minutes}min.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_weather_model",
            )

# ---------------------------------------------------------------------
# 4. СТАН ЗОВНІШНЬОГО ВОЛОГОГО ПОВІТРЯ
# ---------------------------------------------------------------------
with tab_psychro:
    st.subheader("Етап 9. Стан зовнішнього вологого повітря")

    st.write(
        "Розрахунок виконується для кожного часового інтервалу моделі "
        "за температурою зовнішнього повітря `T2M`, відносною вологістю "
        "`RH2M` та атмосферним тиском `PS`."
    )

    st.markdown(
        """
        Визначаються:
        - тиск насиченої водяної пари;
        - парціальний тиск водяної пари;
        - вологовміст повітря;
        - ентальпія вологого повітря;
        - температура точки роси;
        - густина вологого повітря;
        - питомий об'єм на 1 кг сухого повітря.
        """
    )

    if "weather_model" not in st.session_state:
        st.warning(
            "Спочатку відкрийте вкладку «Погодні дані NASA POWER» "
            "та натисніть «Завантажити NASA POWER і сформувати ряд моделі»."
        )
    else:
        weather_model = st.session_state["weather_model"]

        try:
            psychro_df = calculate_psychrometric_state(
                weather_model
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["psychro_model"] = psychro_df

            psychro_summary = summarize_psychrometrics(
                psychro_df
            )

            st.markdown("#### Підсумкові психрометричні показники")
            st.dataframe(
                psychro_summary.style.format(
                    {"Значення": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Середній вологовміст",
                (
                    f"{psychro_df['humidity_ratio_g_kg'].mean():.2f} "
                    "г/кг с.п."
                ),
            )

            p2.metric(
                "Середня ентальпія",
                (
                    f"{psychro_df['enthalpy_kj_kg_da'].mean():.2f} "
                    "кДж/кг с.п."
                ),
            )

            p3.metric(
                "Мін. точка роси",
                f"{psychro_df['dew_point_calc_c'].min():.2f} °C",
            )

            p4.metric(
                "Макс. точка роси",
                f"{psychro_df['dew_point_calc_c'].max():.2f} °C",
            )

            st.markdown("#### Часовий ряд стану зовнішнього повітря")

            display_columns = [
                "operating_mode",
                "T2M",
                "RH2M",
                "PS",
                "p_ws_kpa",
                "p_v_kpa",
                "humidity_ratio_g_kg",
                "enthalpy_kj_kg_da",
                "dew_point_calc_c",
                "T2MDEW",
                "dew_point_error_c",
                "moist_air_density_kg_m3",
                "specific_volume_m3_kg_da",
            ]

            existing_columns = [
                column
                for column in display_columns
                if column in psychro_df.columns
            ]

            st.dataframe(
                psychro_df[existing_columns],
                use_container_width=True,
            )

            st.markdown("#### Вологовміст зовнішнього повітря")
            st.line_chart(
                psychro_df[["humidity_ratio_g_kg"]],
                use_container_width=True,
            )

            st.markdown("#### Ентальпія зовнішнього вологого повітря")
            st.line_chart(
                psychro_df[["enthalpy_kj_kg_da"]],
                use_container_width=True,
            )

            st.markdown("#### Точка роси: розрахунок і NASA POWER")

            dew_columns = ["dew_point_calc_c"]
            if "T2MDEW" in psychro_df.columns:
                dew_columns.append("T2MDEW")

            st.line_chart(
                psychro_df[dew_columns],
                use_container_width=True,
            )

            if "dew_point_abs_error_c" in psychro_df.columns:
                mean_dew_error = (
                    psychro_df["dew_point_abs_error_c"].mean()
                )
                max_dew_error = (
                    psychro_df["dew_point_abs_error_c"].max()
                )

                st.info(
                    "Контрольний розрахунок точки роси: "
                    f"середнє абсолютне відхилення від `T2MDEW` NASA POWER "
                    f"становить {mean_dew_error:.3f} °C, "
                    f"максимальне — {max_dew_error:.3f} °C."
                )

            st.download_button(
                "Завантажити часовий ряд із психрометричними параметрами",
                data=dataframe_to_csv_bytes(psychro_df),
                file_name="weather_psychrometrics.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_psychrometrics",
            )

# ---------------------------------------------------------------------
# 5. ПОТРІБНА ШВИДКІСТЬ ВИДАЛЕННЯ ВОДИ
# ---------------------------------------------------------------------
with tab_removal:
    st.subheader("Етап 10. Визначення потрібної швидкості видалення води")

    st.info(
        "Тут визначається цільова середня швидкість, необхідна для "
        "видалення заданої маси води за сумарний час активного денного "
        "сушіння. Це ще не фактична кінетика сушіння."
    )

    current_product = products.loc[
        products["product_name_uk"] == selected_name
    ].iloc[0]

    current_balance = calculate_mass_balance(
        initial_mass_kg,
        float(current_product["initial_moisture_pct"]),
        float(current_product["final_moisture_pct"]),
    )

    removal_start = datetime.combine(
        process_start_date,
        process_start_time,
        tzinfo=ZoneInfo("UTC"),
    )

    try:
        removal_timeline = build_process_timeline(
            start_local=removal_start,
            duration_hours=process_duration_hours,
            step_minutes=model_step_minutes,
            day_start=day_start_time,
            day_end=day_end_time,
        )

        removal_profile, removal_summary = (
            calculate_required_moisture_removal_profile(
                timeline=removal_timeline,
                water_to_remove_kg=current_balance["water_to_remove_kg"],
                dry_matter_kg=current_balance["dry_matter_kg"],
                initial_water_kg=current_balance["initial_water_kg"],
                step_minutes=model_step_minutes,
            )
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        st.session_state["required_moisture_removal_profile"] = (
            removal_profile
        )

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Вода до видалення",
            f"{current_balance['water_to_remove_kg']:.3f} кг",
        )
        r2.metric(
            "Активне сушіння",
            f"{removal_summary['active_hours']:.1f} год",
        )
        r3.metric(
            "Потрібна середня швидкість",
            f"{removal_summary['average_required_rate_kg_h']:.4f} кг/год",
        )
        r4.metric(
            f"За {model_step_minutes} хв",
            f"{removal_summary['water_per_active_interval_g']:.2f} г",
        )

        st.markdown("#### Розрахункова залежність")

        st.latex(
            r"\overline{\dot m}_{\mathrm{вид}}"
            r"=\frac{m_{\mathrm{вид}}}{\tau_{\mathrm{актив}}}"
        )

        st.latex(
            rf"\overline{{\dot m}}_{{\mathrm{{вид}}}}"
            rf"=\frac{{{current_balance['water_to_remove_kg']:.4f}}}"
            rf"{{{removal_summary['active_hours']:.2f}}}"
            rf"={removal_summary['average_required_rate_kg_h']:.5f}"
            rf"\ \mathrm{{кг/год}}"
        )

        summary_table = summarize_required_moisture_removal(
            water_to_remove_kg=current_balance["water_to_remove_kg"],
            step_minutes=model_step_minutes,
            summary=removal_summary,
        )

        st.dataframe(
            summary_table.style.format({"Значення": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Цільова часова траєкторія")

        columns = [
            "operating_mode",
            "elapsed_hours",
            "required_moisture_removal_rate_kg_h",
            "target_water_removed_interval_kg",
            "target_cumulative_water_removed_kg",
            "target_water_remaining_kg",
            "target_product_mass_kg",
            "target_product_moisture_wb_pct",
            "target_product_moisture_db_kg_kg",
        ]

        st.dataframe(
            removal_profile[columns],
            use_container_width=True,
        )

        st.markdown("#### Цільова швидкість видалення води")
        st.line_chart(
            removal_profile[
                ["required_moisture_removal_rate_kg_h"]
            ],
            use_container_width=True,
        )

        st.markdown("#### Цільове зниження вологості продукту")
        st.line_chart(
            removal_profile[
                ["target_product_moisture_wb_pct"]
            ],
            use_container_width=True,
        )

        st.warning(
            "Ця траєкторія є цільовою: прийнято рівномірне видалення "
            "води у денному режимі та нульове — у нічному. Реальна "
            "швидкість сушіння надалі повинна визначатися кінетичною "
            "моделлю або експериментально."
        )

        st.download_button(
            "Завантажити цільовий профіль видалення води",
            data=dataframe_to_csv_bytes(removal_profile),
            file_name="required_moisture_removal_profile.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_required_moisture_removal",
        )

        st.info(
            "Наступний етап — пункт 11: задання геометричних параметрів "
            "сушильної камери та параметрів шару продукту."
        )

# ---------------------------------------------------------------------
# 6. ГЕОМЕТРІЯ СУШИЛЬНОЇ КАМЕРИ І ШАРУ ПРОДУКТУ
# ---------------------------------------------------------------------
with tab_geometry:
    st.subheader(
        "Етап 11. Геометричні параметри сушильної камери "
        "та параметри шару продукту"
    )

    st.info(
        "На цьому етапі задається лише геометрія установки і шару "
        "продукту. Витрата сушильного агента, стан повітря на виході "
        "та фактична швидкість сушіння поки не визначаються."
    )

    st.markdown("#### Геометрія сушильної камери")

    g1, g2, g3 = st.columns(3)

    with g1:
        chamber_length_m = st.number_input(
            "Внутрішня довжина камери, м",
            min_value=0.10,
            value=1.50,
            step=0.05,
            format="%.3f",
            key="chamber_length_m",
        )

    with g2:
        chamber_width_m = st.number_input(
            "Внутрішня ширина камери, м",
            min_value=0.10,
            value=0.80,
            step=0.05,
            format="%.3f",
            key="chamber_width_m",
        )

    with g3:
        chamber_height_m = st.number_input(
            "Внутрішня висота камери, м",
            min_value=0.10,
            value=1.50,
            step=0.05,
            format="%.3f",
            key="chamber_height_m",
        )

    airflow_direction = st.selectbox(
        "Напрям основного потоку сушильного агента",
        options=[
            "Уздовж довжини камери",
            "Уздовж ширини камери",
            "Вертикально",
        ],
        index=0,
        key="airflow_direction",
    )

    st.markdown("#### Лотки та шар продукту")

    t1, t2, t3 = st.columns(3)

    with t1:
        tray_count = st.number_input(
            "Кількість лотків (ярусів)",
            min_value=1,
            value=5,
            step=1,
            key="tray_count",
        )

    with t2:
        tray_length_m = st.number_input(
            "Корисна довжина одного лотка, м",
            min_value=0.05,
            value=1.20,
            step=0.05,
            format="%.3f",
            key="tray_length_m",
        )

    with t3:
        tray_width_m = st.number_input(
            "Корисна ширина одного лотка, м",
            min_value=0.05,
            value=0.60,
            step=0.05,
            format="%.3f",
            key="tray_width_m",
        )

    # Якщо в Google Sheets для вибраного продукту є рекомендована
    # товщина шару, використовуємо її лише як початкове значення поля.
    sheet_layer_thickness = None
    if "layer_thickness_m" in current_product.index:
        try:
            value = float(current_product["layer_thickness_m"])
            if pd.notna(value) and value > 0:
                sheet_layer_thickness = value
        except (TypeError, ValueError):
            sheet_layer_thickness = None

    default_layer_thickness_m = (
        sheet_layer_thickness
        if sheet_layer_thickness is not None
        else 0.030
    )

    l1, l2 = st.columns(2)

    with l1:
        layer_thickness_m = st.number_input(
            "Товщина шару продукту на лотку, м",
            min_value=0.001,
            value=float(default_layer_thickness_m),
            step=0.005,
            format="%.3f",
            key="layer_thickness_geometry_m",
        )

    with l2:
        tray_pitch_m = st.number_input(
            "Вертикальний крок між ярусами, м",
            min_value=0.01,
            value=0.20,
            step=0.01,
            format="%.3f",
            key="tray_pitch_m",
        )

    reference_bulk_density = None

    if "bulk_density_kg_m3" in current_product.index:
        try:
            bulk_value = float(
                current_product["bulk_density_kg_m3"]
            )
            if pd.notna(bulk_value) and bulk_value > 0:
                reference_bulk_density = bulk_value
        except (TypeError, ValueError):
            reference_bulk_density = None

    if reference_bulk_density is None:
        st.caption(
            "Довідкова насипна густина для вибраного продукту "
            "у Google Sheets не задана. Геометрія буде розрахована "
            "без порівняння з довідковою насипною густиною."
        )
    else:
        st.write(
            "Довідкова насипна густина з Google Sheets: "
            f"**{reference_bulk_density:.1f} кг/м³**."
        )

    try:
        geometry = calculate_dryer_geometry(
            product_mass_kg=float(initial_mass_kg),
            chamber_length_m=float(chamber_length_m),
            chamber_width_m=float(chamber_width_m),
            chamber_height_m=float(chamber_height_m),
            tray_count=int(tray_count),
            tray_length_m=float(tray_length_m),
            tray_width_m=float(tray_width_m),
            layer_thickness_m=float(layer_thickness_m),
            tray_pitch_m=float(tray_pitch_m),
            airflow_direction=airflow_direction,
            reference_bulk_density_kg_m3=(
                reference_bulk_density
            ),
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        st.session_state["dryer_geometry"] = geometry

        st.markdown("#### Розраховані геометричні параметри")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Об'єм камери",
            f"{geometry['chamber_volume_m3']:.3f} м³",
        )

        m2.metric(
            "Сумарна площа сушіння",
            f"{geometry['total_drying_area_m2']:.3f} м²",
        )

        m3.metric(
            "Питоме завантаження",
            f"{geometry['specific_loading_kg_m2']:.2f} кг/м²",
        )

        m4.metric(
            "Геометрична площа перерізу потоку",
            f"{geometry['gross_flow_area_m2']:.3f} м²",
        )

        geometry_table = pd.DataFrame(
            {
                "Параметр": [
                    "Об'єм сушильної камери",
                    "Площа одного лотка",
                    "Сумарна корисна площа сушіння",
                    "Сумарний геометричний об'єм шару продукту",
                    "Питоме завантаження площі сушіння",
                    "Розрахункова об'ємна щільність завантаження",
                    "Частка об'єму шару від об'єму камери",
                    "Геометрична площа поперечного перерізу потоку",
                    "Довжина шляху основного потоку",
                    "Орієнтовна висота пакета ярусів",
                ],
                "Значення": [
                    geometry["chamber_volume_m3"],
                    geometry["tray_area_m2"],
                    geometry["total_drying_area_m2"],
                    geometry["product_layer_volume_m3"],
                    geometry["specific_loading_kg_m2"],
                    geometry["calculated_bulk_loading_kg_m3"],
                    geometry["chamber_fill_fraction_pct"],
                    geometry["gross_flow_area_m2"],
                    geometry["flow_path_length_m"],
                    geometry["required_stack_height_m"],
                ],
                "Одиниця": [
                    "м³",
                    "м²",
                    "м²",
                    "м³",
                    "кг/м²",
                    "кг/м³",
                    "%",
                    "м²",
                    "м",
                    "м",
                ],
            }
        )

        st.dataframe(
            geometry_table.style.format(
                {"Значення": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        if not geometry["tray_fits_in_plan"]:
            st.error(
                "Корисні розміри лотка перевищують внутрішні "
                "розміри камери у плані."
            )
        else:
            st.success(
                "Корисні розміри лотка не перевищують внутрішні "
                "розміри камери у плані."
            )

        if not geometry["stack_fits_height"]:
            st.warning(
                "За заданим кроком між ярусами пакет лотків "
                "перевищує внутрішню висоту камери. Перевірте "
                "кількість ярусів або їх крок."
            )

        if (
            geometry[
                "required_layer_thickness_from_reference_m"
            ]
            is not None
        ):
            st.markdown(
                "#### Перевірка за довідковою насипною густиною"
            )

            st.write(
                "Товщина шару, яка відповідала б заданій масі, "
                "сумарній площі сушіння та довідковій насипній "
                "густині:"
            )

            st.latex(
                r"\delta_{\mathrm{розр}}"
                r"=\frac{m_0}{\rho_{\mathrm{нас}}A_{\mathrm{суш}}}"
            )

            st.metric(
                "Розрахункова товщина шару за ρнас",
                (
                    f"{geometry['required_layer_thickness_from_reference_m']:.4f} "
                    "м"
                ),
            )

            st.caption(
                "Це контроль узгодженості введених геометричних "
                "параметрів, а не автоматична заміна введеної "
                "користувачем товщини шару."
            )

        st.markdown("#### Основні геометричні залежності")

        st.latex(
            r"V_{\mathrm{кам}}=L_{\mathrm{кам}}"
            r"B_{\mathrm{кам}}H_{\mathrm{кам}}"
        )

        st.latex(
            r"A_{\mathrm{суш}}=n_{\mathrm{лот}}"
            r"L_{\mathrm{лот}}B_{\mathrm{лот}}"
        )

        st.latex(
            r"V_{\mathrm{ш}}=A_{\mathrm{суш}}\delta_{\mathrm{ш}}"
        )

        st.latex(
            r"q_A=\frac{m_0}{A_{\mathrm{суш}}}"
        )

        st.info(
            "Наступний етап — пункт 12: вибір і обґрунтування "
            "кінетичної моделі сушіння продукту."
        )

# ---------------------------------------------------------------------
# 7. КІНЕТИЧНА МОДЕЛЬ СУШІННЯ
# ---------------------------------------------------------------------
with tab_kinetics:
    st.subheader(
        "Етап 12. Вибір і обґрунтування кінетичної моделі сушіння"
    )

    st.info(
        "Для пшениці cv. 'Pionier' використовується узагальнена "
        "модель Page за Ramaj et al. (2021). Показник n є сталим, "
        "а коефіцієнт k визначається через температуру, відносну "
        "вологість і швидкість сушильного агента."
    )

    st.markdown("#### Прийнята модель")

    st.success(
        "Для базової моделі сушіння зерна пшениці приймається "
        "напівемпірична модель Page."
    )

    st.latex(
        r"MR=\exp\left(-k\,t^n\right)"
    )

    st.latex(
        r"MR="
        r"\frac{X-X_{\mathrm{e}}}"
        r"{X_0-X_{\mathrm{e}}}"
    )

    st.latex(
        r"k=2.8\cdot10^{-3}"
        r"\exp(0.059T)"
        r"RH^{-0.139}"
        r"v^{0.025}"
    )

    st.latex(r"n=0.784")

    st.markdown("де:")

    st.markdown(
        r"""
- \(MR\) — безрозмірне відношення вологовмісту;
- \(X\) — поточний вологовміст продукту на сухій основі, кг води/кг сухої речовини;
- \(X_0\) — початковий вологовміст продукту на сухій основі, кг води/кг сухої речовини;
- \(X_{\mathrm{e}}\) — рівноважний вологовміст продукту, кг води/кг сухої речовини;
- \(t\) — час сушіння;
- \(k\) — коефіцієнт швидкості сушіння;
- \(n\) — безрозмірний показник моделі Page.
        """
    )

    st.markdown("#### Чому вибрано модель Page")

    justification = pd.DataFrame(
        {
            "Джерело": [
                (
                    "Ramaj et al., 2021, "
                    "Drying Kinetics of Wheat "
                    "(Triticum aestivum L., cv. 'Pionier')"
                ),
                (
                    "Goneli et al., 2007, "
                    "Mathematical Modelling of Thin-Layer "
                    "Drying Kinetics of Wheat"
                ),
            ],
            "Умови дослідження": [
                (
                    "T = 10–50 °C; RH = 20–60 %; "
                    "v = 0,15–1,00 м/с"
                ),
                (
                    "T = 35–55 °C; RH ≈ 55 %"
                ),
            ],
            "Висновок щодо моделі": [
                (
                    "Page показала сприятливу точність опису "
                    "експериментальних кривих сушіння пшениці."
                ),
                (
                    "Серед досліджених тонкошарових моделей "
                    "найкраще узгодження отримано для Page."
                ),
            ],
        }
    )

    st.dataframe(
        justification,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Робочий температурний діапазон нашої моделі 35–45 °C "
        "потрапляє в межі температур, досліджених у наведених "
        "роботах. Це обґрунтовує вибір структури моделі Page, "
        "але не автоматичне перенесення її числових коефіцієнтів."
    )

    st.markdown("#### Статус параметрів моделі")

    parameter_status = pd.DataFrame(
        {
            "Параметр": [
                "X₀",
                "Xₑ",
                "k",
                "n",
            ],
            "Статус": [
                (
                    "Визначається з початкової вологості продукту "
                    "у пунктах 3–4"
                ),
                (
                    "Буде визначено на наступному етапі "
                    "як функцію температури та відносної вологості"
                ),
                (
                    "Визначається за узагальненою залежністю "
                    "Ramaj et al.: k=f(T, RH, v)"
                ),
                (
                    "Для узагальненої моделі прийнято n=0,784 "
                    "за Ramaj et al. (2021)"
                ),
            ],
        }
    )

    st.dataframe(
        parameter_status,
        use_container_width=True,
        hide_index=True,
    )

    # Зберігаємо вибрану структуру моделі для наступних етапів.
    kinetic_model_definition = {
        "model_name": "Page",
        "model_equation": "MR = exp(-k * t**n)",
        "moisture_ratio_equation": (
            "MR = (X - Xe) / (X0 - Xe)"
        ),
        "k_equation": (
            "k = 2.8e-3 * exp(0.059*T) * "
            "RH**(-0.139) * v**0.025"
        ),
        "n": 0.784,
        "equilibrium_moisture_Xe": (
            "Modified Oswin, calculated in stage 13"
        ),
        "parameter_status": (
            "Generalized Page model from Ramaj et al. (2021)."
        ),
    }

    st.session_state["kinetic_model_definition"] = (
        kinetic_model_definition
    )

    st.markdown("#### Послідовність подальшого використання")

    st.write(
        "На цьому етапі програма навмисно не будує фактичну криву "
        "сушіння. Для коректного використання рівняння Page спочатку "
        "необхідно визначити рівноважний вологовміст "
        r"$X_{\mathrm{e}}$, після чого параметризувати "
        r"$k$ і $n$ для відповідних умов сушіння."
    )

    st.warning(
        "Експериментально валідованою для Page є область "
        "T=10–50 °C, RH=20–60 %, v=0,15–1,00 м/с. "
        "У програмі значення k при виході RH за 20–60 % "
        "може бути обчислено як екстраполяція, але такі "
        "інтервали маркуються окремо і не вважаються "
        "експериментально валідованими."
    )

# ---------------------------------------------------------------------
# 8. РІВНОВАЖНА ВОЛОГІСТЬ ПШЕНИЦІ
# ---------------------------------------------------------------------
with tab_equilibrium:
    st.subheader(
        "Етап 13. Визначення рівноважної вологості продукту"
    )

    st.info(
        "Для узгодженості з вибраною моделлю Page використовується "
        "та сама експериментальна робота Ramaj et al. (2021). "
        "Автори визначили рівноважний вологовміст пшениці сорту "
        "Pionier експериментально та апроксимували його моделлю "
        "Modified Oswin."
    )

    st.markdown("#### Модель Modified Oswin")

    st.latex(
        r"X_{\mathrm{eq}}="
        r"\left(C_1+C_2T\right)"
        r"\left("
        r"\frac{\varphi/100}{1-\varphi/100}"
        r"\right)^{1/C_3}"
    )

    st.markdown(
        r"""
де:

- \(X_{\mathrm{eq}}\) — рівноважний вологовміст пшениці на сухій основі, кг води/кг сухої речовини;
- \(T\) — температура сушильного агента, °C;
- \(\varphi\) — відносна вологість сушильного агента, %;
- \(C_1=0{,}129\);
- \(C_2=-6{,}460\cdot10^{-4}\;^\circ\mathrm{C}^{-1}\);
- \(C_3=2{,}944\).
        """
    )

    st.caption(
        "Діапазон експериментальної валідації Ramaj et al. (2021): "
        "10–50 °C та 5,7–86,8 % RH. За межами цього діапазону "
        "програма не виконує екстраполяцію."
    )

    if "psychro_model" not in st.session_state:
        st.warning(
            "Спочатку виконайте етап 9 у вкладці "
            "«Стан вологого повітря», щоб сформувати часовий ряд "
            "вологовмісту та атмосферного тиску."
        )
    else:
        psychro_for_eq = st.session_state["psychro_model"]

        drying_temp_for_eq = float(
            product["target_drying_temp_c"]
        )

        try:
            eq_profile, eq_summary = (
                calculate_equilibrium_moisture_profile(
                    psychro_df=psychro_for_eq,
                    drying_temperature_c=drying_temp_for_eq,
                )
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state[
                "equilibrium_moisture_profile"
            ] = eq_profile

            st.markdown(
                "#### Стан сушильного агента перед контактом із продуктом"
            )

            st.write(
                "У денному режимі повітря приводиться до заданої "
                f"температури сушіння **{drying_temp_for_eq:.1f} °C**. "
                "При чутливому нагріванні без додавання вологи "
                "вологовміст повітря не змінюється:"
            )

            st.latex(
                r"d_{\mathrm{після\ нагр}}="
                r"d_{\mathrm{зовн}}"
            )

            st.write(
                "Тому відносна вологість після нагрівання "
                "визначається заново за незмінного вологовмісту "
                "та поточного атмосферного тиску."
            )

            e1, e2, e3, e4 = st.columns(4)

            e1.metric(
                "Середня RH після нагрівання",
                (
                    f"{eq_summary['mean_drying_air_rh_pct']:.1f} %"
                    if pd.notna(
                        eq_summary["mean_drying_air_rh_pct"]
                    )
                    else "—"
                ),
            )

            e2.metric(
                "Середній Xeq",
                (
                    f"{eq_summary['mean_xeq_db_kg_kg']:.4f} кг/кг с.р."
                    if pd.notna(
                        eq_summary["mean_xeq_db_kg_kg"]
                    )
                    else "—"
                ),
            )

            e3.metric(
                "Середня рівноважна вологість",
                (
                    f"{eq_summary['mean_weq_wb_pct']:.2f} % w.b."
                    if pd.notna(
                        eq_summary["mean_weq_wb_pct"]
                    )
                    else "—"
                ),
            )

            e4.metric(
                "Інтервали поза областю моделі",
                str(eq_summary["invalid_active_intervals"]),
            )

            if eq_summary["invalid_active_intervals"] > 0:
                st.warning(
                    "Для частини активних інтервалів температура "
                    "або RH сушильного агента лежить поза "
                    "експериментальним діапазоном Modified Oswin. "
                    "Для цих інтервалів Xeq залишено порожнім, "
                    "щоб не виконувати необґрунтовану екстраполяцію."
                )
            else:
                st.success(
                    "Усі активні денні інтервали лежать у межах "
                    "експериментальної області Modified Oswin."
                )

            st.markdown("#### Часовий ряд")

            eq_columns = [
                "operating_mode",
                "T2M",
                "RH2M",
                "humidity_ratio_g_kg",
                "PS",
                "drying_air_temperature_c",
                "drying_air_inlet_rh_pct",
                "equilibrium_moisture_db_kg_kg",
                "equilibrium_moisture_wb_pct",
                "oswin_validity_ok",
            ]

            existing_eq_columns = [
                col
                for col in eq_columns
                if col in eq_profile.columns
            ]

            st.dataframe(
                eq_profile[existing_eq_columns],
                use_container_width=True,
            )

            st.markdown(
                "#### Рівноважний вологовміст на сухій основі"
            )

            st.line_chart(
                eq_profile[
                    ["equilibrium_moisture_db_kg_kg"]
                ],
                use_container_width=True,
            )

            st.markdown(
                "#### Рівноважна вологість на вологій основі"
            )

            st.line_chart(
                eq_profile[
                    ["equilibrium_moisture_wb_pct"]
                ],
                use_container_width=True,
            )

            st.warning(
                "Xeq не є фактичною вологістю зерна. Це рівноважне "
                "значення, до якого зерно прагне за тривалого контакту "
                "з повітрям за відповідних T і RH. Фактична зміна "
                "вологовмісту визначатиметься кінетичною моделлю Page."
            )

            st.download_button(
                "Завантажити профіль рівноважної вологості",
                data=dataframe_to_csv_bytes(eq_profile),
                file_name="equilibrium_moisture_profile.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_equilibrium_moisture",
            )

# ---------------------------------------------------------------------
# 9. СИСТЕМА РІВНЯНЬ ТЕПЛО- І МАСООБМІНУ
# ---------------------------------------------------------------------
with tab_heat_mass:
    st.subheader(
        "Етап 14. Система рівнянь тепло- і масообміну "
        "в сушильній камері"
    )

    st.info(
        "На цьому етапі формується математичне ядро сушильної "
        "камери. Повний числовий розв'язок ще не виконується: "
        "не задаються довільно стан повітря на виході, витрата "
        "повітря, коефіцієнт теплообміну або теплофізичні "
        "властивості продукту. Їх буде визначено або обґрунтовано "
        "на наступному етапі."
    )

    st.markdown("#### Система рівнянь")

    st.write(
        "За основу для подальшої моделі приймається одновимірний "
        "нестаціонарний опис спільного перенесення теплоти та "
        "вологи в шарі продукту. Для уникнення плутанини "
        "вологовміст повітря позначено d, а вологовміст продукту — X."
    )

    st.markdown("**1. Баланс енергії сушильного агента**")

    st.latex(
        r"\frac{\partial T_a}{\partial t}"
        r"=-\frac{v_a}{\varepsilon}"
        r"\frac{\partial T_a}{\partial x}"
        r"+"
        r"\frac{h_{a-p}a_s(T_p-T_a)}"
        r"{\varepsilon\rho_a(c_{a,d}+c_v d)}"
    )

    st.markdown("**2. Баланс енергії продукту**")

    st.latex(
        r"\frac{\partial T_p}{\partial t}"
        r"=\frac{1}"
        r"{\rho_{p,d}(c_{p,d}+c_wX)}"
        r"\left["
        r"h_{a-p}a_s(T_a-T_p)"
        r"-\rho_a v_a"
        r"\frac{\partial d}{\partial x}"
        r"\left("
        r"L_v+c_v(T_a-T_p)"
        r"\right)"
        r"\right]"
    )

    st.markdown("**3. Баланс вологи сушильного агента**")

    st.latex(
        r"\frac{\partial d}{\partial t}"
        r"=-\frac{v_a}{\varepsilon}"
        r"\frac{\partial d}{\partial x}"
        r"-\frac{\rho_{p,d}}"
        r"{\varepsilon\rho_a}"
        r"\frac{\partial X}{\partial t}"
    )

    st.markdown("**4. Зміна вологовмісту продукту за Page**")

    st.latex(
        r"\frac{\partial X}{\partial t}"
        r"=-nk\,t_{\mathrm{eff}}^{\,n-1}"
        r"\left(X-X_{\mathrm{eq}}\right)"
    )

    st.markdown("де:")

    st.markdown(
        r"""
- \(T_a\) — температура сушильного агента, °C;
- \(T_p\) — температура продукту, °C;
- \(d\) — вологовміст сушильного агента, кг води/кг сухого повітря;
- \(X\) — вологовміст продукту на сухій основі, кг води/кг сухої речовини;
- \(X_{\mathrm{eq}}\) — рівноважний вологовміст продукту;
- \(v_a\) — швидкість сушильного агента, м/с;
- \(\varepsilon\) — пористість шару;
- \(\rho_a\) — густина вологого повітря, кг/м³;
- \(\rho_{p,d}\) — об'ємна густина сухої речовини продукту, кг сухої речовини/м³ шару;
- \(h_{a-p}\) — коефіцієнт конвективного теплообміну між повітрям і продуктом, Вт/(м²·К);
- \(a_s\) — питома поверхня тепло- і масообміну продукту, м²/м³;
- \(c_{a,d}\) — питома теплоємність сухого повітря, Дж/(кг·К);
- \(c_v\) — питома теплоємність водяної пари, Дж/(кг·К);
- \(c_{p,d}\) — питома теплоємність сухої речовини продукту, Дж/(кг·К);
- \(c_w\) — питома теплоємність води, Дж/(кг·К);
- \(L_v\) — питома теплота випаровування води, Дж/кг;
- \(t_{\mathrm{eff}}\) — ефективний час сушіння, хв.
        """
    )

    st.markdown("#### Параметри Page для поточних умов")

    st.write(
        "Для змінних умов коефіцієнт k визначається на кожному "
        "часовому інтервалі за Ramaj et al. (2021). Інтервали "
        "RH=20–60 % позначаються як валідовані; за межами цього "
        "діапазону k може бути обчислено як явна екстраполяція:"
    )

    st.latex(
        r"k=2.8\cdot10^{-3}"
        r"\exp(0.059T_a)"
        r"RH^{-0.139}"
        r"v_a^{0.025},"
        r"\qquad n=0.784"
    )

    # Try to use a recommended velocity from Google Sheets only as a
    # starting value for the UI. If absent, use the same neutral
    # literature-domain example value used in the methodology example.
    default_velocity = 0.50

    try:
        if (
            "recommended_air_velocity_m_s" in product.index
            and pd.notna(product["recommended_air_velocity_m_s"])
        ):
            candidate_velocity = float(
                product["recommended_air_velocity_m_s"]
            )
            if 0.15 <= candidate_velocity <= 1.00:
                default_velocity = candidate_velocity
    except (TypeError, ValueError):
        pass

    air_velocity_stage14 = st.number_input(
        "Розрахункова швидкість сушильного агента vₐ, м/с",
        min_value=0.15,
        max_value=1.00,
        value=float(default_velocity),
        step=0.05,
        format="%.2f",
        key="stage14_air_velocity_m_s",
    )

    st.caption(
        "Це керований робочий параметр, а не підібраний коефіцієнт "
        "моделі. На наступному етапі швидкість буде пов'язана з "
        "витратою сушильного агента та геометрією камери."
    )

    if "equilibrium_moisture_profile" not in st.session_state:
        st.warning(
            "Спочатку сформуйте результати етапу 13 у вкладці "
            "«Рівноважна вологість»."
        )
    else:
        eq_stage14 = st.session_state[
            "equilibrium_moisture_profile"
        ]

        try:
            coupling_profile, coupling_summary = (
                prepare_heat_mass_transfer_inputs(
                    equilibrium_profile=eq_stage14,
                    air_velocity_m_s=float(air_velocity_stage14),
                )
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state[
                "heat_mass_transfer_input_profile"
            ] = coupling_profile

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Швидкість повітря",
                f"{coupling_summary['air_velocity_m_s']:.2f} м/с",
            )

            c2.metric(
                "Середнє k",
                (
                    f"{coupling_summary['mean_k_min_inv']:.5f} хв⁻¹"
                    if pd.notna(
                        coupling_summary["mean_k_min_inv"]
                    )
                    else "—"
                ),
            )

            c3.metric(
                "Валідовані інтервали",
                (
                    f"{coupling_summary['validated_active_intervals']}"
                    f"/{coupling_summary['active_intervals']}"
                ),
            )

            c4.metric(
                "Екстрапольовані за RH",
                (
                    f"{coupling_summary['extrapolated_active_intervals']}"
                    f"/{coupling_summary['active_intervals']}"
                ),
            )

            if coupling_summary[
                "extrapolated_active_intervals"
            ] > 0:
                st.warning(
                    "Для частини денних інтервалів RH лежить поза "
                    "експериментальним діапазоном Page 20–60 %. "
                    "Коефіцієнт k для них розраховується за тією самою "
                    "кореляцією, але статус інтервалу встановлюється "
                    "як «page_rh_extrapolation». Фактичне RH не "
                    "замінюється штучно на 20 або 60 %."
                )

                s1, s2 = st.columns(2)

                s1.metric(
                    "Середня |Δk| від граничного RH",
                    (
                        f"{coupling_summary['mean_abs_k_extrapolation_difference_pct']:.2f} %"
                    ),
                )

                s2.metric(
                    "Максимальна |Δk| від граничного RH",
                    (
                        f"{coupling_summary['max_abs_k_extrapolation_difference_pct']:.2f} %"
                    ),
                )

                st.caption(
                    "Δk тут є лише діагностикою: порівнюється k за "
                    "фактичним RH з k, який був би отриманий при "
                    "обмеженні RH найближчою межею валідованого "
                    "діапазону (20 або 60 %). У сам розрахунок "
                    "використовується k за фактичним RH."
                )

            if coupling_summary[
                "unusable_active_intervals"
            ] > 0:
                st.error(
                    "Є денні інтервали, для яких повна зв'язка "
                    "Page + Modified Oswin непридатна: температура "
                    "або швидкість лежить поза областю Page, або "
                    "параметри лежать поза областю Modified Oswin."
                )

            if (
                coupling_summary["extrapolated_active_intervals"] == 0
                and coupling_summary["unusable_active_intervals"] == 0
            ):
                st.success(
                    "Усі активні денні інтервали лежать у межах "
                    "експериментальної області Page."
                )

            stage14_columns = [
                "operating_mode",
                "drying_air_temperature_c",
                "drying_air_inlet_rh_pct",
                "air_velocity_m_s",
                "equilibrium_moisture_db_kg_kg",
                "page_k_min_inv",
                "page_n",
                "page_status",
                "drying_model_status",
                "page_validity_ok",
                "page_extrapolated_rh",
                "page_rh_clipped_for_check_pct",
                "page_k_rh_clipped_check_min_inv",
                "page_k_extrapolation_difference_pct",
            ]

            existing_stage14_columns = [
                col
                for col in stage14_columns
                if col in coupling_profile.columns
            ]

            st.markdown(
                "#### Часовий ряд уже визначених параметрів системи"
            )

            st.dataframe(
                coupling_profile[
                    existing_stage14_columns
                ],
                use_container_width=True,
            )

            st.markdown(
                "#### Зміна коефіцієнта k в активному режимі"
            )

            st.line_chart(
                coupling_profile[
                    ["page_k_min_inv"]
                ],
                use_container_width=True,
            )

            st.download_button(
                "Завантажити вхідні параметри тепло- і масообміну",
                data=dataframe_to_csv_bytes(
                    coupling_profile
                ),
                file_name=(
                    "heat_mass_transfer_input_profile.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_heat_mass_inputs",
            )

    st.markdown("#### Що ще потрібно для числового розв'язку")

    missing_parameters = pd.DataFrame(
        {
            "Параметр": [
                "ε",
                "ρp,d",
                "ha-p",
                "as",
                "cp,d",
                "Tp,0",
                "Витрата сушильного агента",
                "Стан повітря на виході",
            ],
            "Призначення": [
                "Пористість шару продукту",
                "Об'ємна густина сухої речовини в шарі",
                "Конвективний теплообмін повітря-продукт",
                "Питома поверхня зерна в шарі",
                "Теплоємність сухої речовини продукту",
                "Початкова температура продукту",
                "Зв'язок швидкості потоку з масовою витратою",
                "Результат спільного балансу повітря і продукту",
            ],
            "Статус": [
                "ще не задано",
                "ще не визначено",
                "ще не визначено",
                "ще не визначено",
                "ще не задано",
                "ще не задано",
                "визначатиметься на етапі 15",
                "визначатиметься на етапі 15",
            ],
        }
    )

    st.dataframe(
        missing_parameters,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Тому на етапі 14 ми лише формуємо систему та обчислюємо "
        "ті її коефіцієнти, які вже мають обґрунтовані вхідні дані. "
        "Повний розрахунок X(t), Tₚ(t), витрати повітря і стану "
        "повітря на виході переноситься на етап 15."
    )

    st.info(
        "Наступний етап — пункт 15: сумісне визначення фактичної "
        "швидкості сушіння, витрати сушильного агента та стану "
        "повітря на виході із сушильної камери."
    )
