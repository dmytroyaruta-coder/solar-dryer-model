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
        # Початкові геометричні параметри зберігаються окремо,
        # щоб наступні етапи могли коректно пов'язувати локальну
        # швидкість повітря в шарі з реальною витратою вентилятора.
        "tray_count": int(tray_count),
        "tray_length_m": float(tray_length_m),
        "tray_width_m": float(tray_width_m),
        "layer_thickness_m": float(layer_thickness_m),
        "tray_pitch_m": float(tray_pitch_m),
        "airflow_direction": airflow_direction,
        "chamber_length_m": float(chamber_length_m),
        "chamber_width_m": float(chamber_width_m),
        "chamber_height_m": float(chamber_height_m),
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


def calculate_dry_air_density_kg_m3(
    temperature_c,
    humidity_ratio_kg_kg,
    pressure_kpa,
):
    """
    Густина саме сухої складової вологого повітря, кг сухого
    повітря/м³ вологого повітря.
    """
    t_k = pd.Series(temperature_c, dtype="float64") + 273.15
    w = pd.Series(humidity_ratio_kg_kg, dtype="float64")
    p = pd.Series(pressure_kpa, dtype="float64")

    p_v = w * p / (0.621945 + w)
    p_da_pa = (p - p_v) * 1000.0

    return p_da_pa / (287.055 * t_k)


def moist_air_enthalpy_kj_kg_da(
    temperature_c,
    humidity_ratio_kg_kg,
):
    """
    Ентальпія вологого повітря, кДж/кг сухого повітря.
    """
    t = pd.Series(temperature_c, dtype="float64")
    w = pd.Series(humidity_ratio_kg_kg, dtype="float64")

    return 1.006 * t + w * (2501.0 + 1.86 * t)


def temperature_from_enthalpy_and_humidity_ratio(
    enthalpy_kj_kg_da,
    humidity_ratio_kg_kg,
):
    """
    Температура вологого повітря з рівняння ентальпії:
        h = 1.006*T + d*(2501 + 1.86*T)

    Використовується лише для діагностичної адіабатичної оцінки
    стану повітря на виході.
    """
    h = pd.Series(enthalpy_kj_kg_da, dtype="float64")
    w = pd.Series(humidity_ratio_kg_kg, dtype="float64")

    return (h - 2501.0 * w) / (1.006 + 1.86 * w)


def saturation_humidity_ratio_kg_kg(
    temperature_c: float,
    pressure_kpa: float,
) -> float:
    """
    Saturation humidity ratio, kg water / kg dry air.
    """
    p_ws = float(
        saturation_vapor_pressure_kpa(
            pd.Series([temperature_c], dtype="float64")
        ).iloc[0]
    )

    if p_ws >= pressure_kpa:
        return float("nan")

    return 0.621945 * p_ws / (pressure_kpa - p_ws)


def moist_air_enthalpy_scalar_kj_kg_da(
    temperature_c: float,
    humidity_ratio_kg_kg: float,
) -> float:
    return (
        1.006 * temperature_c
        + humidity_ratio_kg_kg
        * (2501.0 + 1.86 * temperature_c)
    )


def temperature_from_h_w_scalar(
    enthalpy_kj_kg_da: float,
    humidity_ratio_kg_kg: float,
) -> float:
    return (
        enthalpy_kj_kg_da
        - 2501.0 * humidity_ratio_kg_kg
    ) / (
        1.006
        + 1.86 * humidity_ratio_kg_kg
    )


def relative_humidity_scalar_pct(
    temperature_c: float,
    humidity_ratio_kg_kg: float,
    pressure_kpa: float,
) -> float:
    p_v = (
        humidity_ratio_kg_kg
        * pressure_kpa
        / (0.621945 + humidity_ratio_kg_kg)
    )

    p_ws = float(
        saturation_vapor_pressure_kpa(
            pd.Series([temperature_c], dtype="float64")
        ).iloc[0]
    )

    if p_ws <= 0:
        return float("nan")

    return 100.0 * p_v / p_ws


def equilibrium_moisture_oswin_scalar(
    temperature_c: float,
    relative_humidity_pct: float,
) -> tuple[float, bool]:
    """
    Modified Oswin from Ramaj et al. (2021).

    Returns:
        Xeq, validity
    """
    if not (
        10.0 <= temperature_c <= 50.0
        and 5.7 <= relative_humidity_pct <= 86.8
    ):
        return float("nan"), False

    aw = relative_humidity_pct / 100.0

    if not (0.0 < aw < 1.0):
        return float("nan"), False

    c1 = 0.129
    c2 = -6.460e-4
    c3 = 2.944

    xeq = (
        (c1 + c2 * temperature_c)
        * (aw / (1.0 - aw)) ** (1.0 / c3)
    )

    return float(xeq), True


def page_k_scalar_wheat_pionier(
    temperature_c: float,
    relative_humidity_pct: float,
    air_velocity_m_s: float,
) -> tuple[float, str]:
    """
    Generalized Page correlation from Ramaj et al. (2021).

    Experimental Page conditions:
        T = 10...50 °C
        RH = 20...60 %
        v = 0.15...1.00 m/s

    RH outside 20...60 % is explicitly treated as extrapolation.
    T or v outside their experimental range => no calculation.
    """
    import math

    if not (
        10.0 <= temperature_c <= 50.0
        and 0.15 <= air_velocity_m_s <= 1.00
        and 0.0 < relative_humidity_pct < 100.0
    ):
        return float("nan"), "outside_page_domain"

    k = (
        2.8e-3
        * math.exp(0.059 * temperature_c)
        * relative_humidity_pct ** (-0.139)
        * air_velocity_m_s ** 0.025
    )

    if 20.0 <= relative_humidity_pct <= 60.0:
        status = "validated"
    else:
        status = "rh_extrapolation"

    return float(k), status


def adiabatic_saturation_state(
    inlet_enthalpy_kj_kg_da: float,
    pressure_kpa: float,
    upper_temperature_c: float,
) -> tuple[float, float]:
    """
    Finds the saturated state on the same moist-air enthalpy line.

    It is used only as a physical upper bound on how much additional
    water can remain in the vapour phase during the present simplified
    stage-15 calculation.

    Returns:
        T_sat_ad, w_sat_ad
    """
    import math

    upper = min(float(upper_temperature_c), 60.0)
    lower = -30.0

    def residual(temp_c):
        w_sat = saturation_humidity_ratio_kg_kg(
            temp_c,
            pressure_kpa,
        )

        if not math.isfinite(w_sat):
            return float("nan")

        return (
            moist_air_enthalpy_scalar_kj_kg_da(
                temp_c,
                w_sat,
            )
            - inlet_enthalpy_kj_kg_da
        )

    f_low = residual(lower)
    f_high = residual(upper)

    if not (
        math.isfinite(f_low)
        and math.isfinite(f_high)
    ):
        return float("nan"), float("nan")

    # For normal unsaturated inlet air, the constant-enthalpy
    # saturation state should lie between lower and inlet T.
    if f_low > 0 or f_high < 0:
        return float("nan"), float("nan")

    for _ in range(70):
        mid = 0.5 * (lower + upper)
        f_mid = residual(mid)

        if not math.isfinite(f_mid):
            return float("nan"), float("nan")

        if f_mid > 0:
            upper = mid
        else:
            lower = mid

    t_sat = 0.5 * (lower + upper)
    w_sat = saturation_humidity_ratio_kg_kg(
        t_sat,
        pressure_kpa,
    )

    return float(t_sat), float(w_sat)


def simulate_zonal_coupled_drying_stage15(
    input_profile: pd.DataFrame,
    dry_matter_kg: float,
    initial_x_db: float,
    target_x_db: float,
    step_minutes: int,
    active_flow_area_per_path_m2: float,
    parallel_path_count: int,
    zone_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Improved stage 15 with a physically explicit linkage:

        v_Page  <->  active product-flow area  <->  fan flow

    Ramaj et al. (2021) measured drying-air velocity in a column
    upstream of a sample holder with a perforated bottom so that air
    flowed through the pore volume of the wheat kernels. Therefore,
    v used in the Page correlation is treated here as a LOCAL
    superficial velocity through the representative product-flow path,
    not as an automatic velocity through the full chamber cross-section.

    For one representative path:
        Vdot_path = v_Page * A_path

    For N identical parallel paths:
        Vdot_fan = N * Vdot_path

    The representative path contains:
        m_dry,path = m_dry,total / N

    The air is propagated serially through computational zones inside
    that representative path. Results are then scaled by N identical
    parallel paths.

    This allows:
    - one serial air path through all trays: N = 1;
    - parallel air distribution between trays: N = number of trays;
    - any other justified number of identical parallel branches.

    No arbitrary "chamber cross-section = Page area" assumption is used.
    """
    import math
    import numpy as np

    required = {
        "operating_mode",
        "drying_air_temperature_c",
        "humidity_ratio_kg_kg",
        "PS",
        "air_velocity_m_s",
    }

    missing = required - set(input_profile.columns)
    if missing:
        raise ValueError(
            "Для етапу 15 відсутні параметри: "
            + ", ".join(sorted(missing))
        )

    if dry_matter_kg <= 0:
        raise ValueError(
            "Маса сухої речовини має бути більшою за нуль."
        )

    if target_x_db >= initial_x_db:
        raise ValueError(
            "Кінцевий вологовміст має бути меншим за початковий."
        )

    if active_flow_area_per_path_m2 <= 0:
        raise ValueError(
            "Активна площа проходу через продукт має бути > 0."
        )

    if parallel_path_count < 1:
        raise ValueError(
            "Кількість паралельних повітряних трактів має бути >= 1."
        )

    if zone_count < 2:
        raise ValueError(
            "Для зональної моделі потрібно щонайменше 2 зони."
        )

    dt_min = float(step_minutes)
    dt_s = dt_min * 60.0
    dt_h = dt_min / 60.0

    result = input_profile.copy()
    n_rows = len(result)

    n_parallel = int(parallel_path_count)
    area_path = float(active_flow_area_per_path_m2)
    area_total = area_path * n_parallel

    # Representative path carries an equal fraction of the product.
    dry_mass_path = float(dry_matter_kg) / n_parallel
    dry_mass_zone = dry_mass_path / int(zone_count)

    # Each parallel path is assumed identical, so one path can be
    # simulated and then multiplied by N for total water removal.
    x_zone = np.full(
        int(zone_count),
        float(initial_x_db),
        dtype=float,
    )

    tau_zone_min = np.zeros(
        int(zone_count),
        dtype=float,
    )

    source_initial_x_db = 0.159
    source_initial_x_uncertainty = 0.001
    source_upper_observed_x_db = (
        source_initial_x_db
        + source_initial_x_uncertainty
    )

    avg_x_start = np.full(n_rows, np.nan)
    avg_x_end = np.full(n_rows, np.nan)
    avg_wb_end = np.full(n_rows, np.nan)

    total_removed_interval = np.zeros(n_rows)
    removal_rate_kg_h = np.zeros(n_rows)

    total_air_volume_flow_m3_h = np.zeros(n_rows)
    path_air_volume_flow_m3_h = np.zeros(n_rows)

    total_dry_air_mass_flow_kg_h = np.zeros(n_rows)
    path_dry_air_mass_flow_kg_h = np.zeros(n_rows)

    inlet_w = np.full(n_rows, np.nan)
    outlet_w = np.full(n_rows, np.nan)

    inlet_t = np.full(n_rows, np.nan)
    outlet_t = np.full(n_rows, np.nan)

    inlet_rh = np.full(n_rows, np.nan)
    outlet_rh = np.full(n_rows, np.nan)

    active_drying = np.zeros(n_rows, dtype=bool)
    all_zones_target = np.zeros(n_rows, dtype=bool)

    capacity_limited_interval = np.zeros(
        n_rows,
        dtype=bool,
    )

    source_moisture_extrapolation_interval = np.zeros(
        n_rows,
        dtype=bool,
    )

    page_rh_extrapolation_interval = np.zeros(
        n_rows,
        dtype=bool,
    )

    unusable_zone_interval = np.zeros(
        n_rows,
        dtype=bool,
    )

    zone_records = []

    source_extrapolated_removed_kg = 0.0
    total_capacity_limited_kg = 0.0

    for pos in range(n_rows):
        row = result.iloc[pos]

        avg_x_start[pos] = float(x_zone.mean())

        is_day = (
            row["operating_mode"] == "Денний режим"
        )

        zones_finished_before = bool(
            np.all(
                x_zone <= float(target_x_db) + 1e-12
            )
        )

        can_run = (
            is_day
            and not zones_finished_before
            and pd.notna(row["drying_air_temperature_c"])
            and pd.notna(row["humidity_ratio_kg_kg"])
            and pd.notna(row["PS"])
            and pd.notna(row["air_velocity_m_s"])
        )

        if not can_run:
            avg_x_end[pos] = float(x_zone.mean())
            avg_wb_end[pos] = (
                avg_x_end[pos]
                / (1.0 + avg_x_end[pos])
                * 100.0
            )

            all_zones_target[pos] = bool(
                np.all(
                    x_zone
                    <= float(target_x_db) + 1e-12
                )
            )
            continue

        active_drying[pos] = True

        t_air = float(
            row["drying_air_temperature_c"]
        )
        w_air = float(
            row["humidity_ratio_kg_kg"]
        )
        p_kpa = float(row["PS"])

        # This is the LOCAL Page velocity from stage 14.
        velocity_page = float(row["air_velocity_m_s"])

        inlet_t[pos] = t_air
        inlet_w[pos] = w_air
        inlet_rh[pos] = relative_humidity_scalar_pct(
            t_air,
            w_air,
            p_kpa,
        )

        # ---------------------------------------------------------
        # Correct v_Page <-> area <-> fan flow linkage
        # ---------------------------------------------------------
        volume_flow_path_m3_s = (
            velocity_page * area_path
        )

        volume_flow_total_m3_s = (
            volume_flow_path_m3_s * n_parallel
        )

        rho_da_in = float(
            calculate_dry_air_density_kg_m3(
                temperature_c=pd.Series([t_air]),
                humidity_ratio_kg_kg=pd.Series([w_air]),
                pressure_kpa=pd.Series([p_kpa]),
            ).iloc[0]
        )

        mdot_da_path_kg_s = (
            rho_da_in * volume_flow_path_m3_s
        )

        mdot_da_total_kg_s = (
            mdot_da_path_kg_s * n_parallel
        )

        if (
            not math.isfinite(mdot_da_path_kg_s)
            or mdot_da_path_kg_s <= 0
        ):
            raise ValueError(
                "Не вдалося визначити додатну масову "
                "витрату сухого повітря в одному тракті."
            )

        path_air_volume_flow_m3_h[pos] = (
            volume_flow_path_m3_s * 3600.0
        )

        total_air_volume_flow_m3_h[pos] = (
            volume_flow_total_m3_s * 3600.0
        )

        path_dry_air_mass_flow_kg_h[pos] = (
            mdot_da_path_kg_s * 3600.0
        )

        total_dry_air_mass_flow_kg_h[pos] = (
            mdot_da_total_kg_s * 3600.0
        )

        removed_path_this_step = 0.0

        # Air propagates serially through zones of ONE representative
        # path. Parallel paths are assumed identical.
        for zone_idx in range(int(zone_count)):
            x_old = float(x_zone[zone_idx])

            zone_in_t = float(t_air)
            zone_in_w = float(w_air)

            zone_in_rh = relative_humidity_scalar_pct(
                zone_in_t,
                zone_in_w,
                p_kpa,
            )

            xeq, oswin_ok = (
                equilibrium_moisture_oswin_scalar(
                    zone_in_t,
                    zone_in_rh,
                )
            )

            k, page_status = (
                page_k_scalar_wheat_pionier(
                    zone_in_t,
                    zone_in_rh,
                    velocity_page,
                )
            )

            source_moisture_extrapolation = (
                x_old
                > source_upper_observed_x_db
                + 1e-12
            )

            if source_moisture_extrapolation:
                source_moisture_extrapolation_interval[
                    pos
                ] = True

            if page_status == "rh_extrapolation":
                page_rh_extrapolation_interval[pos] = True

            zone_finished = (
                x_old
                <= float(target_x_db) + 1e-12
            )

            model_usable = (
                not zone_finished
                and oswin_ok
                and math.isfinite(k)
                and math.isfinite(xeq)
                and x_old > xeq
            )

            page_removed_path_kg = 0.0
            air_capacity_path_kg = 0.0
            actual_removed_path_kg = 0.0
            limited_by_air_capacity = False

            if model_usable:
                n_page = 0.784

                tau_old = float(
                    tau_zone_min[zone_idx]
                )

                exponent_increment = (
                    k
                    * (
                        (tau_old + dt_min) ** n_page
                        - tau_old ** n_page
                    )
                )

                x_page = (
                    xeq
                    + (x_old - xeq)
                    * math.exp(-exponent_increment)
                )

                x_page = max(
                    float(target_x_db),
                    float(x_page),
                )

                page_removed_path_kg = max(
                    0.0,
                    dry_mass_zone
                    * (x_old - x_page),
                )

                h_zone_in = (
                    moist_air_enthalpy_scalar_kj_kg_da(
                        zone_in_t,
                        zone_in_w,
                    )
                )

                _, w_sat_ad = adiabatic_saturation_state(
                    inlet_enthalpy_kj_kg_da=h_zone_in,
                    pressure_kpa=p_kpa,
                    upper_temperature_c=zone_in_t,
                )

                if math.isfinite(w_sat_ad):
                    air_capacity_path_kg = max(
                        0.0,
                        mdot_da_path_kg_s
                        * dt_s
                        * (w_sat_ad - zone_in_w),
                    )
                else:
                    air_capacity_path_kg = float("inf")

                actual_removed_path_kg = min(
                    page_removed_path_kg,
                    air_capacity_path_kg,
                )

                limited_by_air_capacity = (
                    actual_removed_path_kg
                    < page_removed_path_kg - 1e-12
                )

                if limited_by_air_capacity:
                    capacity_limited_interval[pos] = True

                    # Convert the lost capacity to total dryer scale.
                    total_capacity_limited_kg += (
                        page_removed_path_kg
                        - actual_removed_path_kg
                    ) * n_parallel

                x_new = (
                    x_old
                    - actual_removed_path_kg
                    / dry_mass_zone
                )

                x_new = max(
                    float(target_x_db),
                    float(x_new),
                )

                tau_zone_min[zone_idx] += dt_min

            else:
                x_new = x_old

                if (
                    not zone_finished
                    and not model_usable
                ):
                    unusable_zone_interval[pos] = True

            x_zone[zone_idx] = x_new

            if (
                source_moisture_extrapolation
                and actual_removed_path_kg > 0
            ):
                source_extrapolated_removed_kg += (
                    actual_removed_path_kg
                    * n_parallel
                )

            removed_path_this_step += (
                actual_removed_path_kg
            )

            zone_out_w = (
                zone_in_w
                + actual_removed_path_kg
                / (
                    mdot_da_path_kg_s
                    * dt_s
                )
            )

            h_zone_in = (
                moist_air_enthalpy_scalar_kj_kg_da(
                    zone_in_t,
                    zone_in_w,
                )
            )

            zone_out_t = (
                temperature_from_h_w_scalar(
                    h_zone_in,
                    zone_out_w,
                )
            )

            zone_out_rh = (
                relative_humidity_scalar_pct(
                    zone_out_t,
                    zone_out_w,
                    p_kpa,
                )
            )

            zone_records.append(
                {
                    "time_local": (
                        result.iloc[pos]["time_local"]
                        if "time_local" in result.columns
                        else result.index[pos]
                    ),
                    "time_position": pos,
                    "zone": zone_idx + 1,
                    "zone_count": int(zone_count),
                    "parallel_path_count": n_parallel,
                    "active_flow_area_per_path_m2": area_path,
                    "total_active_flow_area_m2": area_total,
                    "page_velocity_m_s": velocity_page,
                    "fan_flow_path_m3_h": (
                        volume_flow_path_m3_s * 3600.0
                    ),
                    "fan_flow_total_m3_h": (
                        volume_flow_total_m3_s * 3600.0
                    ),
                    "x_start_db_kg_kg": x_old,
                    "x_end_db_kg_kg": x_new,
                    "moisture_end_wb_pct": (
                        x_new / (1.0 + x_new) * 100.0
                    ),
                    "air_inlet_temp_c": zone_in_t,
                    "air_inlet_rh_pct": zone_in_rh,
                    "air_inlet_humidity_ratio_g_kg": (
                        zone_in_w * 1000.0
                    ),
                    "xeq_db_kg_kg": xeq,
                    "page_k_min_inv": k,
                    "page_status": page_status,
                    "oswin_validity_ok": oswin_ok,
                    "source_initial_moisture_extrapolation": (
                        source_moisture_extrapolation
                    ),
                    "page_predicted_water_removed_path_kg": (
                        page_removed_path_kg
                    ),
                    "air_vapour_capacity_path_kg": (
                        air_capacity_path_kg
                        if math.isfinite(air_capacity_path_kg)
                        else float("nan")
                    ),
                    "actual_water_removed_path_kg": (
                        actual_removed_path_kg
                    ),
                    "actual_water_removed_total_kg": (
                        actual_removed_path_kg
                        * n_parallel
                    ),
                    "air_capacity_limited": (
                        limited_by_air_capacity
                    ),
                    "air_outlet_temp_c": zone_out_t,
                    "air_outlet_rh_pct": zone_out_rh,
                    "air_outlet_humidity_ratio_g_kg": (
                        zone_out_w * 1000.0
                    ),
                }
            )

            t_air = zone_out_t
            w_air = zone_out_w

        removed_total_this_step = (
            removed_path_this_step * n_parallel
        )

        total_removed_interval[pos] = (
            removed_total_this_step
        )

        removal_rate_kg_h[pos] = (
            removed_total_this_step / dt_h
        )

        outlet_t[pos] = t_air
        outlet_w[pos] = w_air
        outlet_rh[pos] = (
            relative_humidity_scalar_pct(
                t_air,
                w_air,
                p_kpa,
            )
        )

        avg_x_end[pos] = float(x_zone.mean())
        avg_wb_end[pos] = (
            avg_x_end[pos]
            / (1.0 + avg_x_end[pos])
            * 100.0
        )

        all_zones_target[pos] = bool(
            np.all(
                x_zone
                <= float(target_x_db) + 1e-12
            )
        )

    result["zonal_average_x_start_db_kg_kg"] = avg_x_start
    result["zonal_average_x_end_db_kg_kg"] = avg_x_end

    result["zonal_average_moisture_end_wb_pct"] = (
        avg_wb_end
    )

    result["actual_water_removed_interval_kg"] = (
        total_removed_interval
    )

    result["actual_water_removal_rate_kg_h"] = (
        removal_rate_kg_h
    )

    result["active_drying_command"] = active_drying
    result["all_zones_target_reached"] = (
        all_zones_target
    )

    result["page_reference_flow_area_per_path_m2"] = area_path
    result["parallel_air_path_count"] = n_parallel
    result["page_reference_flow_area_total_m2"] = area_total

    result["fan_volume_flow_per_path_m3_h"] = (
        path_air_volume_flow_m3_h
    )

    result["fan_volume_flow_total_m3_h"] = (
        total_air_volume_flow_m3_h
    )

    result["dry_air_mass_flow_per_path_kg_h"] = (
        path_dry_air_mass_flow_kg_h
    )

    result["dry_air_mass_flow_total_kg_h"] = (
        total_dry_air_mass_flow_kg_h
    )

    result["zonal_air_inlet_temp_c"] = inlet_t
    result["zonal_air_outlet_temp_c"] = outlet_t

    result["zonal_air_inlet_rh_pct"] = inlet_rh
    result["zonal_air_outlet_rh_pct"] = outlet_rh

    result["zonal_air_inlet_humidity_ratio_kg_kg"] = (
        inlet_w
    )

    result["zonal_air_outlet_humidity_ratio_kg_kg"] = (
        outlet_w
    )

    result["zonal_air_outlet_humidity_ratio_g_kg"] = (
        outlet_w * 1000.0
    )

    result["air_capacity_limited_interval"] = (
        capacity_limited_interval
    )

    result[
        "source_initial_moisture_extrapolation_interval"
    ] = source_moisture_extrapolation_interval

    result[
        "page_rh_extrapolation_interval"
    ] = page_rh_extrapolation_interval

    result["unusable_zone_interval"] = (
        unusable_zone_interval
    )

    zone_df = pd.DataFrame(zone_records)

    final_avg_x = float(x_zone.mean())
    final_avg_wb = (
        final_avg_x
        / (1.0 + final_avg_x)
        * 100.0
    )

    total_removed = float(
        total_removed_interval.sum()
    )

    positive_rates = result.loc[
        result["actual_water_removal_rate_kg_h"] > 0,
        "actual_water_removal_rate_kg_h",
    ]

    positive_air_total = result.loc[
        result["fan_volume_flow_total_m3_h"] > 0,
        "fan_volume_flow_total_m3_h",
    ]

    positive_air_path = result.loc[
        result["fan_volume_flow_per_path_m3_h"] > 0,
        "fan_volume_flow_per_path_m3_h",
    ]

    target_reached = bool(
        np.all(
            x_zone <= float(target_x_db) + 1e-12
        )
    )

    target_time = None

    if result["all_zones_target_reached"].any():
        first_target_pos = int(
            np.flatnonzero(
                result[
                    "all_zones_target_reached"
                ].to_numpy()
            )[0]
        )

        if "time_local" in result.columns:
            try:
                target_time_start = pd.to_datetime(
                    result.iloc[first_target_pos]["time_local"]
                )
                target_time = str(
                    target_time_start
                    + pd.Timedelta(minutes=int(step_minutes))
                )
            except Exception:
                target_time = str(
                    result.iloc[first_target_pos]["time_local"]
                )
        else:
            try:
                target_time_start = pd.to_datetime(
                    result.index[first_target_pos]
                )
                target_time = str(
                    target_time_start
                    + pd.Timedelta(minutes=int(step_minutes))
                )
            except Exception:
                target_time = str(
                    result.index[first_target_pos]
                )

    summary = {
        "source_initial_x_db": source_initial_x_db,
        "source_initial_x_uncertainty": (
            source_initial_x_uncertainty
        ),
        "source_upper_observed_x_db": (
            source_upper_observed_x_db
        ),
        "source_initial_wb_pct": (
            source_initial_x_db
            / (1.0 + source_initial_x_db)
            * 100.0
        ),
        "model_initial_x_db": float(initial_x_db),
        "model_initial_wb_pct": (
            float(initial_x_db)
            / (1.0 + float(initial_x_db))
            * 100.0
        ),
        "initial_moisture_extrapolation": bool(
            initial_x_db > source_upper_observed_x_db
        ),
        "zone_count": int(zone_count),
        "parallel_path_count": n_parallel,
        "active_flow_area_per_path_m2": area_path,
        "active_flow_area_total_m2": area_total,
        "final_average_x_db": final_avg_x,
        "final_average_wb_pct": final_avg_wb,
        "total_water_removed_kg": total_removed,
        "mean_actual_removal_rate_kg_h": (
            float(positive_rates.mean())
            if not positive_rates.empty
            else 0.0
        ),
        "max_actual_removal_rate_kg_h": (
            float(positive_rates.max())
            if not positive_rates.empty
            else 0.0
        ),
        "mean_fan_flow_total_m3_h": (
            float(positive_air_total.mean())
            if not positive_air_total.empty
            else 0.0
        ),
        "mean_fan_flow_per_path_m3_h": (
            float(positive_air_path.mean())
            if not positive_air_path.empty
            else 0.0
        ),
        "target_reached": target_reached,
        "target_time": target_time,
        "source_extrapolated_removed_kg": (
            float(source_extrapolated_removed_kg)
        ),
        "source_extrapolated_removed_fraction_pct": (
            100.0
            * source_extrapolated_removed_kg
            / total_removed
            if total_removed > 0
            else 0.0
        ),
        "air_capacity_limited_intervals": int(
            capacity_limited_interval.sum()
        ),
        "air_capacity_limited_water_difference_kg": (
            float(total_capacity_limited_kg)
        ),
        "page_rh_extrapolated_intervals": int(
            page_rh_extrapolation_interval.sum()
        ),
        "unusable_intervals": int(
            unusable_zone_interval.sum()
        ),
    }

    return result, zone_df, summary


def evaluate_drying_performance_stage16(
    coupled_profile: pd.DataFrame,
    target_profile: pd.DataFrame,
    water_to_remove_kg: float,
    step_minutes: int,
) -> tuple[pd.DataFrame, dict[str, float | int | bool | str | None]]:
    """
    Пункт 16. Перевірка забезпечення потрібної швидкості
    видалення води.

    Порівнюються:
    - фактична зональна кінетика з етапу 15;
    - цільова траєкторія з етапу 10.

    Важливо:
    миттєва швидкість після дострокового досягнення кінцевої
    вологості не трактується як "дефіцит", оскільки після виконання
    технологічної цілі активне сушіння вже не потрібне.

    Основним критерієм є накопичувальне виконання графіка:
        ΔmΣ(t) = mфакт,Σ(t) - mціль,Σ(t)

    Якщо ΔmΣ < 0 до моменту досягнення цілі, модель відстає від
    цільового графіка.
    """
    import numpy as np

    required_actual = {
        "actual_water_removed_interval_kg",
        "actual_water_removal_rate_kg_h",
        "zonal_average_moisture_end_wb_pct",
        "all_zones_target_reached",
    }

    required_target = {
        "target_water_removed_interval_kg",
        "required_moisture_removal_rate_kg_h",
        "target_cumulative_water_removed_kg",
        "target_product_moisture_wb_pct",
        "operating_mode",
    }

    missing_actual = required_actual - set(coupled_profile.columns)
    missing_target = required_target - set(target_profile.columns)

    if missing_actual:
        raise ValueError(
            "В етапі 15 відсутні колонки: "
            + ", ".join(sorted(missing_actual))
        )

    if missing_target:
        raise ValueError(
            "У цільовому профілі етапу 10 відсутні колонки: "
            + ", ".join(sorted(missing_target))
        )

    if len(coupled_profile) != len(target_profile):
        raise ValueError(
            "Фактичний і цільовий часові ряди мають різну "
            "кількість інтервалів."
        )

    if water_to_remove_kg <= 0:
        raise ValueError(
            "Маса води до видалення повинна бути більшою за нуль."
        )

    dt_h = float(step_minutes) / 60.0
    tolerance_kg = max(1e-9, water_to_remove_kg * 1e-6)

    result = pd.DataFrame(index=coupled_profile.index)

    if "time_local" in coupled_profile.columns:
        result["time_local"] = coupled_profile["time_local"].to_numpy()

    result["operating_mode"] = (
        target_profile["operating_mode"].to_numpy()
    )

    result["required_rate_kg_h"] = pd.to_numeric(
        target_profile["required_moisture_removal_rate_kg_h"],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    result["actual_rate_kg_h"] = pd.to_numeric(
        coupled_profile["actual_water_removal_rate_kg_h"],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    result["rate_difference_kg_h"] = (
        result["actual_rate_kg_h"]
        - result["required_rate_kg_h"]
    )

    result["target_removed_interval_kg"] = pd.to_numeric(
        target_profile["target_water_removed_interval_kg"],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    result["actual_removed_interval_kg"] = pd.to_numeric(
        coupled_profile["actual_water_removed_interval_kg"],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    result["target_cumulative_removed_kg"] = pd.to_numeric(
        target_profile["target_cumulative_water_removed_kg"],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    result["actual_cumulative_removed_kg"] = (
        result["actual_removed_interval_kg"]
        .cumsum()
        .clip(upper=water_to_remove_kg)
    )

    result["cumulative_margin_kg"] = (
        result["actual_cumulative_removed_kg"]
        - result["target_cumulative_removed_kg"]
    )

    result["target_moisture_wb_pct"] = pd.to_numeric(
        target_profile["target_product_moisture_wb_pct"],
        errors="coerce",
    ).to_numpy()

    result["actual_moisture_wb_pct"] = pd.to_numeric(
        coupled_profile["zonal_average_moisture_end_wb_pct"],
        errors="coerce",
    ).to_numpy()

    target_reached_series = (
        coupled_profile["all_zones_target_reached"]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    result["actual_target_reached"] = target_reached_series

    # "Відставання" перевіряємо лише до фактичного виконання цілі.
    if target_reached_series.any():
        first_reached_pos = int(
            np.flatnonzero(target_reached_series)[0]
        )
    else:
        first_reached_pos = len(result) - 1

    before_or_at_completion = (
        np.arange(len(result)) <= first_reached_pos
    )

    result["behind_target_schedule"] = (
        (
            result["cumulative_margin_kg"].to_numpy()
            < -tolerance_kg
        )
        & before_or_at_completion
    )

    result["schedule_status"] = "випередження/виконання"

    result.loc[
        result["behind_target_schedule"],
        "schedule_status",
    ] = "відставання"

    result.loc[
        result["actual_target_reached"],
        "schedule_status",
    ] = "кінцева вологість досягнута"

    active_target_mask = (
        result["required_rate_kg_h"] > 0
    )

    active_actual_mask = (
        result["actual_removed_interval_kg"] > tolerance_kg
    )

    required_active_hours = (
        int(active_target_mask.sum()) * dt_h
    )

    actual_active_hours = (
        int(active_actual_mask.sum()) * dt_h
    )

    required_average_rate = (
        water_to_remove_kg / required_active_hours
        if required_active_hours > 0
        else float("nan")
    )

    actual_total_removed = float(
        result["actual_removed_interval_kg"].sum()
    )

    actual_average_rate = (
        actual_total_removed / actual_active_hours
        if actual_active_hours > 0
        else 0.0
    )

    productivity_factor = (
        actual_average_rate / required_average_rate
        if required_average_rate > 0
        else float("nan")
    )

    productivity_reserve_pct = (
        (productivity_factor - 1.0) * 100.0
        if np.isfinite(productivity_factor)
        else float("nan")
    )

    active_time_reserve_h = (
        required_active_hours - actual_active_hours
    )

    completion_criterion_met = bool(
        actual_total_removed
        >= water_to_remove_kg - tolerance_kg
    )

    behind_intervals = int(
        result["behind_target_schedule"].sum()
    )

    minimum_cumulative_margin = float(
        result.loc[
            before_or_at_completion,
            "cumulative_margin_kg",
        ].min()
    )

    maximum_cumulative_lead = float(
        result["cumulative_margin_kg"].max()
    )

    # Completion clock time:
    # row timestamp is the START of the last calculation interval,
    # therefore add Δt to obtain the end of the interval.
    completion_time = None

    if completion_criterion_met and target_reached_series.any():
        completion_row = coupled_profile.iloc[first_reached_pos]

        if "time_local" in coupled_profile.columns:
            try:
                completion_start = pd.to_datetime(
                    completion_row["time_local"]
                )
                completion_time = str(
                    completion_start
                    + pd.Timedelta(minutes=int(step_minutes))
                )
            except Exception:
                completion_time = None
        else:
            try:
                completion_start = pd.to_datetime(
                    coupled_profile.index[first_reached_pos]
                )
                completion_time = str(
                    completion_start
                    + pd.Timedelta(minutes=int(step_minutes))
                )
            except Exception:
                completion_time = None

    summary = {
        "water_to_remove_kg": float(water_to_remove_kg),
        "actual_total_removed_kg": actual_total_removed,
        "required_active_hours": float(required_active_hours),
        "actual_active_hours": float(actual_active_hours),
        "required_average_rate_kg_h": float(required_average_rate),
        "actual_average_rate_kg_h": float(actual_average_rate),
        "productivity_factor": float(productivity_factor),
        "productivity_reserve_pct": float(productivity_reserve_pct),
        "active_time_reserve_h": float(active_time_reserve_h),
        "completion_criterion_met": completion_criterion_met,
        "completion_time": completion_time,
        "behind_schedule_intervals": behind_intervals,
        "minimum_cumulative_margin_kg": minimum_cumulative_margin,
        "maximum_cumulative_lead_kg": maximum_cumulative_lead,
        "schedule_never_behind_until_completion": (
            behind_intervals == 0
        ),
    }

    return result, summary


def calculate_air_heating_demand_stage17(
    coupled_profile: pd.DataFrame,
    step_minutes: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """
    Пункт 17. Теплота на нагрівання сушильного агента.

    Розраховується теплова потужність, необхідна лише для
    чутливого нагрівання зовнішнього вологого повітря до заданої
    температури сушіння перед входом у сушильну камеру.

    Перед нагріванням і після нього вологовміст d вважається
    незмінним:

        d_heat,out = d_amb

    Ентальпія вологого повітря:
        h = 1.006*T + d*(2501 + 1.86*T)
        [кДж/кг сухого повітря]

    Потрібна теплова потужність:
        Qdot_air = mdot_da * (h_dry - h_amb)

    де mdot_da береться у кг сухого повітря/с.
    Результат Qdot_air отримується у кВт.

    Важливо:
    це лише теплота на нагрівання сушильного агента.
    Тут ще НЕ враховані:
    - теплота випаровування вологи;
    - нагрівання продукту;
    - нагрівання конструкцій;
    - теплові втрати;
    - рекуперація.

    Температура сушіння поки є заданою цільовою температурою.
    Реальна можливість забезпечити її сонячним колектором,
    акумулятором та ТЕН буде перевірятися на наступних етапах.
    """
    import numpy as np

    required = {
        "T2M",
        "humidity_ratio_kg_kg",
        "drying_air_temperature_c",
        "dry_air_mass_flow_total_kg_h",
        "active_drying_command",
    }

    missing = required - set(coupled_profile.columns)
    if missing:
        raise ValueError(
            "Для етапу 17 відсутні параметри: "
            + ", ".join(sorted(missing))
        )

    if step_minutes <= 0:
        raise ValueError(
            "Крок моделювання має бути більшим за нуль."
        )

    result = coupled_profile.copy()

    t_amb = pd.to_numeric(
        result["T2M"],
        errors="coerce",
    )

    d = pd.to_numeric(
        result["humidity_ratio_kg_kg"],
        errors="coerce",
    )

    t_dry = pd.to_numeric(
        result["drying_air_temperature_c"],
        errors="coerce",
    )

    mdot_da_kg_h = pd.to_numeric(
        result["dry_air_mass_flow_total_kg_h"],
        errors="coerce",
    ).fillna(0.0)

    active = (
        result["active_drying_command"]
        .fillna(False)
        .astype(bool)
    )

    # Enthalpy before and after sensible heating.
    h_amb = moist_air_enthalpy_kj_kg_da(
        temperature_c=t_amb,
        humidity_ratio_kg_kg=d,
    )
    h_amb.index = result.index

    h_dry = moist_air_enthalpy_kj_kg_da(
        temperature_c=t_dry,
        humidity_ratio_kg_kg=d,
    )
    h_dry.index = result.index

    delta_h = h_dry - h_amb

    # Equivalent humid-air specific heat per kg dry air.
    cp_moist_air = 1.006 + 1.86 * d

    delta_t = t_dry - t_amb

    mdot_da_kg_s = mdot_da_kg_h / 3600.0

    # Signed thermal power. Positive => heating required.
    qdot_signed_kw = (
        mdot_da_kg_s * delta_h
    )

    # Only active drying intervals require this heating in the
    # current stage. Negative values (ambient hotter than target)
    # are not counted as heating demand.
    qdot_heating_kw = qdot_signed_kw.where(
        active,
        0.0,
    )

    qdot_heating_kw = qdot_heating_kw.clip(
        lower=0.0
    )

    qdot_cooling_equivalent_kw = (
        (-qdot_signed_kw)
        .where(active, 0.0)
        .clip(lower=0.0)
    )

    dt_h = float(step_minutes) / 60.0

    heat_energy_kwh_interval = (
        qdot_heating_kw * dt_h
    )

    cumulative_heat_energy_kwh = (
        heat_energy_kwh_interval.cumsum()
    )

    # Diagnostic cross-check using cp*DeltaT.
    qdot_cp_check_kw = (
        mdot_da_kg_s
        * cp_moist_air
        * delta_t
    ).where(
        active,
        0.0,
    )

    qdot_cp_check_kw = qdot_cp_check_kw.clip(
        lower=0.0
    )

    check_difference_kw = (
        qdot_heating_kw
        - qdot_cp_check_kw
    )

    result["air_heating_ambient_temp_c"] = t_amb
    result["air_heating_target_temp_c"] = t_dry
    result["air_heating_delta_t_c"] = delta_t

    result["air_heating_humidity_ratio_kg_kg"] = d

    result["air_heating_cp_kj_kg_da_k"] = (
        cp_moist_air
    )

    result["air_heating_h_ambient_kj_kg_da"] = (
        h_amb
    )

    result["air_heating_h_target_kj_kg_da"] = (
        h_dry
    )

    result["air_heating_delta_h_kj_kg_da"] = (
        delta_h
    )

    result["air_heating_dry_air_mass_flow_kg_h"] = (
        mdot_da_kg_h
    )

    result["air_heating_power_kw"] = (
        qdot_heating_kw
    )

    result["air_cooling_equivalent_power_kw"] = (
        qdot_cooling_equivalent_kw
    )

    result["air_heating_energy_kwh_interval"] = (
        heat_energy_kwh_interval
    )

    result["air_heating_energy_kwh_cumulative"] = (
        cumulative_heat_energy_kwh
    )

    result["air_heating_cp_check_kw"] = (
        qdot_cp_check_kw
    )

    result["air_heating_balance_check_kw"] = (
        check_difference_kw
    )

    active_heating = result.loc[
        result["air_heating_power_kw"] > 0
    ]

    total_heat_kwh = float(
        result["air_heating_energy_kwh_interval"].sum()
    )

    peak_heat_kw = (
        float(
            active_heating[
                "air_heating_power_kw"
            ].max()
        )
        if not active_heating.empty
        else 0.0
    )

    mean_heat_kw = (
        float(
            active_heating[
                "air_heating_power_kw"
            ].mean()
        )
        if not active_heating.empty
        else 0.0
    )

    mean_delta_t = (
        float(
            active_heating[
                "air_heating_delta_t_c"
            ].mean()
        )
        if not active_heating.empty
        else 0.0
    )

    mean_delta_h = (
        float(
            active_heating[
                "air_heating_delta_h_kj_kg_da"
            ].mean()
        )
        if not active_heating.empty
        else 0.0
    )

    active_heating_hours = (
        int((result["air_heating_power_kw"] > 0).sum())
        * dt_h
    )

    cooling_intervals = int(
        (
            result["air_cooling_equivalent_power_kw"]
            > 0
        ).sum()
    )

    max_balance_error_kw = float(
        result["air_heating_balance_check_kw"]
        .abs()
        .max()
    )

    specific_heat_per_removed_water = float("nan")

    if (
        "actual_water_removed_interval_kg"
        in result.columns
    ):
        removed_water = float(
            pd.to_numeric(
                result["actual_water_removed_interval_kg"],
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )

        if removed_water > 0:
            specific_heat_per_removed_water = (
                total_heat_kwh / removed_water
            )

    summary = {
        "total_air_heating_energy_kwh": (
            total_heat_kwh
        ),
        "peak_air_heating_power_kw": (
            peak_heat_kw
        ),
        "mean_air_heating_power_kw": (
            mean_heat_kw
        ),
        "active_air_heating_hours": (
            float(active_heating_hours)
        ),
        "mean_air_heating_delta_t_c": (
            mean_delta_t
        ),
        "mean_air_heating_delta_h_kj_kg_da": (
            mean_delta_h
        ),
        "cooling_intervals": cooling_intervals,
        "max_enthalpy_cp_check_error_kw": (
            max_balance_error_kw
        ),
        "specific_air_heating_kwh_per_kg_removed_water": (
            specific_heat_per_removed_water
        ),
    }

    return result, summary


def calculate_evaporation_energy_stage18(
    coupled_profile: pd.DataFrame,
    zone_profile: pd.DataFrame,
    step_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """
    Пункт 18. Енергія, пов'язана з фазовим переходом води.

    Для кожної зони:
        h_g(T)  ≈ 2501 + 1.86*T       [кДж/кг води]
        h_f(T)  ≈ 4.186*T             [кДж/кг води]
        h_fg(T) = h_g - h_f
                ≈ 2501 - 2.326*T      [кДж/кг води]

    Як тимчасову температуру фазового переходу до розрахунку
    температури продукту (п.19) використовуємо середню температуру
    сушильного агента в межах відповідної зони:

        T_ref = (T_air,in + T_air,out)/2

    Тоді:
        Q_evap,zone = Δm_w * h_fg(T_ref)

    і середня потужність за часовий крок:
        Qdot_evap = Q_evap / Δt

    ВАЖЛИВО:
    Q_evap на цьому етапі є окремою енергетичною характеристикою
    фазового переходу. Вона НЕ додається автоматично до теплоти
    попереднього нагрівання повітря з п.17, оскільки в зональній
    моделі випаровування вже супроводжується зміною ентальпійного
    стану сушильного агента. Остаточний зовнішній тепловий баланс
    буде сформовано після розрахунку всіх складових.

    Після появи T_product(t) у п.19 цей розрахунок слід уточнити,
    використавши температуру продукту/поверхні замість T_ref.
    """
    import numpy as np

    required_main = {
        "actual_water_removed_interval_kg",
        "active_drying_command",
    }

    required_zone = {
        "time_position",
        "zone",
        "actual_water_removed_total_kg",
        "air_inlet_temp_c",
        "air_outlet_temp_c",
    }

    missing_main = required_main - set(coupled_profile.columns)
    missing_zone = required_zone - set(zone_profile.columns)

    if missing_main:
        raise ValueError(
            "Для етапу 18 у часовому профілі відсутні параметри: "
            + ", ".join(sorted(missing_main))
        )

    if missing_zone:
        raise ValueError(
            "Для етапу 18 у зональному профілі відсутні параметри: "
            + ", ".join(sorted(missing_zone))
        )

    if step_minutes <= 0:
        raise ValueError(
            "Крок моделювання має бути більшим за нуль."
        )

    zone_result = zone_profile.copy()

    zone_result["evaporation_reference_temp_c"] = (
        pd.to_numeric(
            zone_result["air_inlet_temp_c"],
            errors="coerce",
        )
        + pd.to_numeric(
            zone_result["air_outlet_temp_c"],
            errors="coerce",
        )
    ) / 2.0

    t_ref = zone_result["evaporation_reference_temp_c"]

    # Approximate liquid-water enthalpy relative to 0 °C.
    zone_result["liquid_water_enthalpy_kj_kg"] = (
        4.186 * t_ref
    )

    # ASHRAE approximate water-vapour enthalpy relative to 0 °C.
    zone_result["water_vapour_enthalpy_kj_kg"] = (
        2501.0 + 1.86 * t_ref
    )

    zone_result["latent_heat_vaporization_kj_kg"] = (
        zone_result["water_vapour_enthalpy_kj_kg"]
        - zone_result["liquid_water_enthalpy_kj_kg"]
    )

    removed_zone_kg = pd.to_numeric(
        zone_result["actual_water_removed_total_kg"],
        errors="coerce",
    ).fillna(0.0)

    zone_result["evaporation_energy_kj"] = (
        removed_zone_kg
        * zone_result["latent_heat_vaporization_kj_kg"]
    )

    zone_result["evaporation_energy_kwh"] = (
        zone_result["evaporation_energy_kj"] / 3600.0
    )

    dt_s = float(step_minutes) * 60.0

    zone_result["evaporation_power_kw"] = (
        zone_result["evaporation_energy_kj"] / dt_s
    )

    # -----------------------------------------------------------------
    # Aggregate all zones to the model time step.
    # -----------------------------------------------------------------
    grouped = (
        zone_result
        .groupby("time_position", sort=True)
        .agg(
            evaporation_water_removed_kg=(
                "actual_water_removed_total_kg",
                "sum",
            ),
            evaporation_energy_kj=(
                "evaporation_energy_kj",
                "sum",
            ),
            evaporation_energy_kwh=(
                "evaporation_energy_kwh",
                "sum",
            ),
        )
    )

    # Water-removal-weighted reference temperature and h_fg.
    weighted_rows = []

    for time_position, group in zone_result.groupby(
        "time_position",
        sort=True,
    ):
        weights = pd.to_numeric(
            group["actual_water_removed_total_kg"],
            errors="coerce",
        ).fillna(0.0)

        weight_sum = float(weights.sum())

        if weight_sum > 0:
            weighted_t = float(
                (
                    group["evaporation_reference_temp_c"]
                    * weights
                ).sum()
                / weight_sum
            )

            weighted_hfg = float(
                (
                    group["latent_heat_vaporization_kj_kg"]
                    * weights
                ).sum()
                / weight_sum
            )
        else:
            weighted_t = float("nan")
            weighted_hfg = float("nan")

        weighted_rows.append(
            {
                "time_position": int(time_position),
                "evaporation_reference_temp_c": weighted_t,
                "latent_heat_vaporization_kj_kg": weighted_hfg,
            }
        )

    weighted_df = (
        pd.DataFrame(weighted_rows)
        .set_index("time_position")
        if weighted_rows
        else pd.DataFrame()
    )

    grouped = grouped.join(
        weighted_df,
        how="left",
    )

    grouped["evaporation_power_kw"] = (
        grouped["evaporation_energy_kj"] / dt_s
    )

    result = coupled_profile.copy()

    result["time_position"] = range(len(result))

    grouped_for_merge = grouped.reset_index()

    result = result.merge(
        grouped_for_merge,
        on="time_position",
        how="left",
        suffixes=("", "_stage18"),
    )

    # Restore the real local-time axis after merge(), because pandas
    # otherwise replaces the datetime index with a numeric RangeIndex.
    if "time_local" in result.columns:
        result["time_local"] = pd.to_datetime(
            result["time_local"],
            errors="coerce",
        )
        if result["time_local"].notna().any():
            result = result.set_index(
                "time_local",
                drop=False,
            )

    numeric_fill_zero = [
        "evaporation_water_removed_kg",
        "evaporation_energy_kj",
        "evaporation_energy_kwh",
        "evaporation_power_kw",
    ]

    for col in numeric_fill_zero:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce",
        ).fillna(0.0)

    result["evaporation_energy_kwh_cumulative"] = (
        result["evaporation_energy_kwh"].cumsum()
    )

    # Cross-check against water removed in the main stage-15 profile.
    result["evaporation_water_balance_error_kg"] = (
        result["evaporation_water_removed_kg"]
        - pd.to_numeric(
            result["actual_water_removed_interval_kg"],
            errors="coerce",
        ).fillna(0.0)
    )

    total_removed_kg = float(
        result["evaporation_water_removed_kg"].sum()
    )

    total_evap_kwh = float(
        result["evaporation_energy_kwh"].sum()
    )

    positive = result.loc[
        result["evaporation_water_removed_kg"] > 0
    ]

    peak_evap_power_kw = (
        float(positive["evaporation_power_kw"].max())
        if not positive.empty
        else 0.0
    )

    mean_evap_power_kw = (
        float(positive["evaporation_power_kw"].mean())
        if not positive.empty
        else 0.0
    )

    weighted_mean_hfg = (
        float(
            zone_result["evaporation_energy_kj"].sum()
            / total_removed_kg
        )
        if total_removed_kg > 0
        else float("nan")
    )

    weights_all = pd.to_numeric(
        zone_result["actual_water_removed_total_kg"],
        errors="coerce",
    ).fillna(0.0)

    weighted_mean_tref = (
        float(
            (
                zone_result["evaporation_reference_temp_c"]
                * weights_all
            ).sum()
            / weights_all.sum()
        )
        if float(weights_all.sum()) > 0
        else float("nan")
    )

    specific_evap_kwh_per_kg = (
        total_evap_kwh / total_removed_kg
        if total_removed_kg > 0
        else float("nan")
    )

    max_water_balance_error_kg = float(
        result["evaporation_water_balance_error_kg"]
        .abs()
        .max()
    )

    summary = {
        "total_water_evaporated_kg": total_removed_kg,
        "total_evaporation_energy_kwh": total_evap_kwh,
        "peak_evaporation_power_kw": peak_evap_power_kw,
        "mean_evaporation_power_kw": mean_evap_power_kw,
        "weighted_mean_reference_temp_c": weighted_mean_tref,
        "weighted_mean_latent_heat_kj_kg": weighted_mean_hfg,
        "specific_evaporation_energy_kwh_per_kg": (
            specific_evap_kwh_per_kg
        ),
        "max_water_balance_error_kg": (
            max_water_balance_error_kg
        ),
    }

    return result, zone_result, summary


def wheat_moist_specific_heat_kj_kg_k(
    moisture_wb_fraction,
    dry_matter_cp_kj_kg_k: float = 1.550,
    water_cp_kj_kg_k: float = 4.186,
):
    """
    Specific heat capacity of moist wheat as a mass-weighted mixture:

        cp,wheat = (1 - w) * cp,dry + w * cp,water

    where w is moisture fraction on wet basis.

    The dry-matter value cp,dry = 1.550 kJ/(kg·K) is taken from:
    Kharchenko Y., Chornyi V., Sharan A. (2020),
    "Influence of water temperature and moisture increase on wheat
    temperature during moistening", Ukrainian Journal of Food Science,
    8(1), 58–67, DOI 10.24263/2310-1008-2020-8-1-7.

    Water cp is taken as 4.186 kJ/(kg·K) in the current temperature
    range.

    This function gives a thermophysical property; it does NOT solve
    the transient temperature of the grain.
    """
    import numpy as np

    w = pd.Series(
        moisture_wb_fraction,
        dtype="float64",
    )

    w = w.clip(lower=0.0, upper=1.0)

    return (
        (1.0 - w) * float(dry_matter_cp_kj_kg_k)
        + w * float(water_cp_kj_kg_k)
    )


def calculate_product_sensible_heat_stage19(
    coupled_profile: pd.DataFrame,
    dry_matter_kg: float,
    initial_total_mass_kg: float,
    initial_x_db: float,
    target_x_db: float,
    initial_product_temp_c: float,
    reference_product_temp_c: float,
    active_drying_hours: float,
    dry_matter_cp_kj_kg_k: float = 1.550,
    water_cp_kj_kg_k: float = 4.186,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Stage 19: sensible heat associated with warming the product.

    This stage does NOT invent a transient T_product(t), because the
    full product-temperature equation requires heat-transfer data
    (h_a-p, a_s, etc.) not yet justified.

    Instead, it calculates a transparent design heat requirement for
    warming the initial wet product from T_p,0 to a selected reference
    product temperature T_p,ref:

        Q_p,sens =
            [m_dry * cp,dry + m_w,0 * cp,w] * (T_p,ref - T_p,0)

    This is equivalent to:
        Q_p,sens = m_0 * cp,wet,0 * DeltaT

    The result is decomposed into:
    - dry matter sensible heat;
    - sensible heat of water that remains in the final product;
    - sensible heat of the water that is later evaporated.

    The latter is useful because h_fg in stage 18 contains latent heat
    at the phase-change temperature, but not the sensible heating of
    liquid water from the initial product temperature up to that
    reference temperature.

    IMPORTANT:
    This stage produces an energy requirement / design estimate,
    not a validated dynamic product-temperature trajectory.
    """
    import numpy as np

    if dry_matter_kg <= 0:
        raise ValueError(
            "Маса сухої речовини має бути більшою за нуль."
        )

    if initial_total_mass_kg <= 0:
        raise ValueError(
            "Початкова маса продукту має бути більшою за нуль."
        )

    if target_x_db < 0 or initial_x_db < 0:
        raise ValueError(
            "Вологовміст продукту не може бути від'ємним."
        )

    delta_t = (
        float(reference_product_temp_c)
        - float(initial_product_temp_c)
    )

    if delta_t < 0:
        raise ValueError(
            "Референсна температура продукту має бути не нижчою "
            "за його початкову температуру."
        )

    water_initial_kg = (
        float(dry_matter_kg)
        * float(initial_x_db)
    )

    water_final_kg = (
        float(dry_matter_kg)
        * float(target_x_db)
    )

    water_removed_kg = max(
        0.0,
        water_initial_kg - water_final_kg,
    )

    final_product_mass_kg = (
        float(dry_matter_kg)
        + water_final_kg
    )

    w0_wb = (
        water_initial_kg
        / float(initial_total_mass_kg)
    )

    wf_wb = (
        water_final_kg
        / final_product_mass_kg
    )

    cp_initial = float(
        wheat_moist_specific_heat_kj_kg_k(
            pd.Series([w0_wb]),
            dry_matter_cp_kj_kg_k=dry_matter_cp_kj_kg_k,
            water_cp_kj_kg_k=water_cp_kj_kg_k,
        ).iloc[0]
    )

    cp_final = float(
        wheat_moist_specific_heat_kj_kg_k(
            pd.Series([wf_wb]),
            dry_matter_cp_kj_kg_k=dry_matter_cp_kj_kg_k,
            water_cp_kj_kg_k=water_cp_kj_kg_k,
        ).iloc[0]
    )

    q_dry_kj = (
        float(dry_matter_kg)
        * float(dry_matter_cp_kj_kg_k)
        * delta_t
    )

    q_water_initial_kj = (
        water_initial_kg
        * float(water_cp_kj_kg_k)
        * delta_t
    )

    q_water_final_kj = (
        water_final_kg
        * float(water_cp_kj_kg_k)
        * delta_t
    )

    q_removed_water_sensible_kj = (
        water_removed_kg
        * float(water_cp_kj_kg_k)
        * delta_t
    )

    q_total_kj = (
        q_dry_kj
        + q_water_initial_kj
    )

    q_total_kwh = (
        q_total_kj / 3600.0
    )

    q_final_product_only_kj = (
        q_dry_kj
        + q_water_final_kj
    )

    q_final_product_only_kwh = (
        q_final_product_only_kj / 3600.0
    )

    q_removed_water_sensible_kwh = (
        q_removed_water_sensible_kj / 3600.0
    )

    equivalent_mean_power_kw = (
        q_total_kwh / float(active_drying_hours)
        if active_drying_hours > 0
        else float("nan")
    )

    # -----------------------------------------------------------------
    # Property profile from the actual moisture history of stage 15.
    # This does not assume T_product(t); it only tracks how cp changes
    # as water is removed.
    # -----------------------------------------------------------------
    profile = coupled_profile.copy()

    if "zonal_average_moisture_end_wb_pct" not in profile.columns:
        raise ValueError(
            "Для етапу 19 відсутня фактична вологість продукту "
            "з етапу 15."
        )

    moisture_wb_fraction = (
        pd.to_numeric(
            profile["zonal_average_moisture_end_wb_pct"],
            errors="coerce",
        )
        / 100.0
    )

    profile[
        "product_moisture_wb_fraction_stage19"
    ] = moisture_wb_fraction

    profile[
        "product_specific_heat_kj_kg_k"
    ] = wheat_moist_specific_heat_kj_kg_k(
        moisture_wb_fraction,
        dry_matter_cp_kj_kg_k=dry_matter_cp_kj_kg_k,
        water_cp_kj_kg_k=water_cp_kj_kg_k,
    )

    x_actual = pd.to_numeric(
        profile["zonal_average_x_end_db_kg_kg"],
        errors="coerce",
    )

    profile[
        "product_water_mass_kg_stage19"
    ] = (
        float(dry_matter_kg) * x_actual
    )

    profile[
        "product_total_mass_kg_stage19"
    ] = (
        float(dry_matter_kg)
        + profile["product_water_mass_kg_stage19"]
    )

    profile[
        "product_initial_temperature_c_stage19"
    ] = float(initial_product_temp_c)

    profile[
        "product_reference_temperature_c_stage19"
    ] = float(reference_product_temp_c)

    # -----------------------------------------------------------------
    # Sensitivity of design sensible heat to the selected reference
    # product temperature.
    # -----------------------------------------------------------------
    if delta_t > 0:
        t_values = np.linspace(
            float(initial_product_temp_c),
            float(reference_product_temp_c),
            41,
        )
    else:
        t_values = np.array(
            [float(initial_product_temp_c)]
        )

    sensitivity = pd.DataFrame(
        {
            "product_reference_temp_c": t_values,
        }
    )

    sensitivity["delta_t_c"] = (
        sensitivity["product_reference_temp_c"]
        - float(initial_product_temp_c)
    )

    sensitivity[
        "product_sensible_heat_kwh"
    ] = (
        (
            float(dry_matter_kg)
            * float(dry_matter_cp_kj_kg_k)
            + water_initial_kg
            * float(water_cp_kj_kg_k)
        )
        * sensitivity["delta_t_c"]
        / 3600.0
    )

    summary = {
        "initial_product_temp_c": (
            float(initial_product_temp_c)
        ),
        "reference_product_temp_c": (
            float(reference_product_temp_c)
        ),
        "delta_t_c": float(delta_t),
        "dry_matter_cp_kj_kg_k": (
            float(dry_matter_cp_kj_kg_k)
        ),
        "water_cp_kj_kg_k": (
            float(water_cp_kj_kg_k)
        ),
        "initial_wet_product_cp_kj_kg_k": (
            cp_initial
        ),
        "final_wet_product_cp_kj_kg_k": (
            cp_final
        ),
        "water_initial_kg": (
            float(water_initial_kg)
        ),
        "water_final_kg": (
            float(water_final_kg)
        ),
        "water_removed_kg": (
            float(water_removed_kg)
        ),
        "dry_matter_sensible_heat_kwh": (
            q_dry_kj / 3600.0
        ),
        "initial_water_sensible_heat_kwh": (
            q_water_initial_kj / 3600.0
        ),
        "final_product_sensible_heat_kwh": (
            q_final_product_only_kwh
        ),
        "removed_water_sensible_heat_kwh": (
            q_removed_water_sensible_kwh
        ),
        "total_initial_product_sensible_heat_kwh": (
            q_total_kwh
        ),
        "equivalent_mean_power_kw": (
            float(equivalent_mean_power_kw)
        ),
        "active_drying_hours": (
            float(active_drying_hours)
        ),
    }

    return profile, sensitivity, summary



def calculate_construction_thermal_capacity_stage20(
    components_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """
    Пункт 20. Теплова ємність конструкцій сушильної камери.

    Використовуються тільки підтверджувані параметри:
        m_i, c_p,i
    або
        m_i = rho_i * A_i * delta_i

    Для кожного елемента:
        C_i = m_i * c_p,i

    Сумарна теплова ємність:
        C_constr = sum(C_i)        [кДж/К]

    На цьому етапі НЕ задаються:
    - T_ref конструкції;
    - середня температура теплоізоляції;
    - тривалість нагрівання конструкції;
    - реальна теплота або потужність нагрівання.

    Пізніше, після отримання температурної динаміки:
        dQ_constr = sum(m_i*c_p,i*dT_i)
        Qdot_constr = sum(m_i*c_p,i*dT_i/dt)
    """

    required = {
        "include",
        "component",
        "mass_kg",
        "cp_kj_kg_k",
    }

    missing = required - set(components_df.columns)
    if missing:
        raise ValueError(
            "Для етапу 20 відсутні колонки: "
            + ", ".join(sorted(missing))
        )

    result = components_df.copy()

    result["include"] = (
        result["include"]
        .fillna(False)
        .astype(bool)
    )

    result["mass_kg"] = pd.to_numeric(
        result["mass_kg"],
        errors="coerce",
    )

    result["cp_kj_kg_k"] = pd.to_numeric(
        result["cp_kj_kg_k"],
        errors="coerce",
    )

    active = result["include"]

    if not active.any():
        raise ValueError(
            "Не вибрано жодного елемента конструкції. "
            "Увімкніть тільки ті елементи, для яких уже "
            "підтверджені маса/геометрія та cₚ матеріалу."
        )

    invalid = (
        active
        & (
            result["mass_kg"].isna()
            | (result["mass_kg"] <= 0)
            | result["cp_kj_kg_k"].isna()
            | (result["cp_kj_kg_k"] <= 0)
        )
    )

    if invalid.any():
        bad_names = (
            result.loc[invalid, "component"]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Для активних елементів повинні бути "
            "підтверджені m > 0 та cₚ > 0: "
            + ", ".join(bad_names)
        )

    result["thermal_capacity_kj_k"] = (
        result["mass_kg"]
        * result["cp_kj_kg_k"]
    )

    result.loc[
        ~result["include"],
        "thermal_capacity_kj_k",
    ] = 0.0

    result["energy_per_1k_kwh"] = (
        result["thermal_capacity_kj_k"]
        / 3600.0
    )

    total_mass_kg = float(
        result.loc[
            result["include"],
            "mass_kg",
        ].sum()
    )

    total_capacity_kj_k = float(
        result["thermal_capacity_kj_k"].sum()
    )

    energy_per_1k_kwh = (
        total_capacity_kj_k / 3600.0
    )

    active_result = result.loc[
        result["include"]
    ].copy()

    if total_capacity_kj_k > 0:
        active_result["capacity_share_pct"] = (
            active_result["thermal_capacity_kj_k"]
            / total_capacity_kj_k
            * 100.0
        )
    else:
        active_result["capacity_share_pct"] = 0.0

    result["capacity_share_pct"] = 0.0
    result.loc[
        active_result.index,
        "capacity_share_pct",
    ] = active_result["capacity_share_pct"]

    weighted_cp_kj_kg_k = (
        float(
            (
                active_result["mass_kg"]
                * active_result["cp_kj_kg_k"]
            ).sum()
            / total_mass_kg
        )
        if total_mass_kg > 0
        else float("nan")
    )

    dominant_component = "—"
    dominant_share_pct = 0.0

    if not active_result.empty:
        dominant_idx = (
            active_result[
                "thermal_capacity_kj_k"
            ].idxmax()
        )
        dominant_component = str(
            active_result.loc[
                dominant_idx,
                "component",
            ]
        )
        dominant_share_pct = float(
            active_result.loc[
                dominant_idx,
                "capacity_share_pct",
            ]
        )

    summary = {
        "active_component_count": int(
            active_result.shape[0]
        ),
        "total_construction_mass_kg": total_mass_kg,
        "total_thermal_capacity_kj_k": total_capacity_kj_k,
        "energy_per_1k_kwh": energy_per_1k_kwh,
        "weighted_cp_kj_kg_k": weighted_cp_kj_kg_k,
        "dominant_component": dominant_component,
        "dominant_share_pct": dominant_share_pct,
    }

    return result, summary



def upgrade_stage20_components_for_linkage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds the stage-20/21 linkage fields to an older saved table
    without deleting previously entered values.
    """
    out = df.copy()

    defaults = {
        "element_role": "Внутрішній елемент",
        "assembly": "",
        "area_m2": 0.0,
        "thickness_mm": 0.0,
        "density_kg_m3": 0.0,
        "cp_kj_kg_k": 0.0,
        "thermal_conductivity_w_m_k": 0.0,
        "source_note": "",
    }

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    # Migration of the names used in the previous version.
    component_text = (
        out["component"]
        .astype(str)
        .str.lower()
    )

    envelope_mask = (
        component_text.str.contains("корпус", na=False)
        | component_text.str.contains("теплоізоля", na=False)
        | component_text.str.contains("обшив", na=False)
        | component_text.str.contains("стін", na=False)
    )

    out.loc[
        envelope_mask
        & (
            out["element_role"].astype(str)
            == "Внутрішній елемент"
        ),
        "element_role",
    ] = "Шар огородження"

    out.loc[
        envelope_mask
        & (
            out["assembly"].astype(str).str.strip() == ""
        ),
        "assembly",
    ] = "Основне огородження"

    return out


def build_envelope_assemblies_stage21(
    construction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forms envelope assemblies directly from the saved stage-20 table.

    Only active rows with element_role == "Шар огородження"
    participate in the wall thermal-resistance calculation.

    For each assembly:
        R_layers = sum(delta_i / lambda_i)
        U_layers = 1 / R_layers

    U_layers is NOT the final air-to-air U-value because surface
    convective resistances are not yet included.
    """
    required = {
        "include",
        "component",
        "element_role",
        "assembly",
        "area_m2",
        "thickness_mm",
        "thermal_conductivity_w_m_k",
    }

    missing = required - set(construction_df.columns)
    if missing:
        raise ValueError(
            "Для зв'язку п.20→п.21 відсутні колонки: "
            + ", ".join(sorted(missing))
        )

    layers = construction_df.copy()

    layers["include"] = (
        layers["include"]
        .fillna(False)
        .astype(bool)
    )

    layers = layers.loc[
        layers["include"]
        & (
            layers["element_role"].astype(str)
            == "Шар огородження"
        )
    ].copy()

    if layers.empty:
        raise ValueError(
            "У п.20 не позначено жодного активного рядка "
            "як «Шар огородження»."
        )

    layers["assembly"] = (
        layers["assembly"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if (layers["assembly"] == "").any():
        bad = layers.loc[
            layers["assembly"] == "",
            "component",
        ].astype(str).tolist()

        raise ValueError(
            "Для шарів огородження потрібно вказати вузол "
            "(наприклад «Основне огородження»): "
            + ", ".join(bad)
        )

    for col in [
        "area_m2",
        "thickness_mm",
        "thermal_conductivity_w_m_k",
    ]:
        layers[col] = pd.to_numeric(
            layers[col],
            errors="coerce",
        )

    invalid = (
        layers["area_m2"].isna()
        | (layers["area_m2"] <= 0)
        | layers["thickness_mm"].isna()
        | (layers["thickness_mm"] <= 0)
        | layers["thermal_conductivity_w_m_k"].isna()
        | (layers["thermal_conductivity_w_m_k"] <= 0)
    )

    if invalid.any():
        bad = layers.loc[
            invalid,
            "component",
        ].astype(str).tolist()

        raise ValueError(
            "Для розрахунку п.21 кожний активний шар огородження "
            "повинен мати A>0, δ>0 та λ>0: "
            + ", ".join(bad)
        )

    layers["thickness_m"] = (
        layers["thickness_mm"] / 1000.0
    )

    layers["layer_resistance_m2_k_w"] = (
        layers["thickness_m"]
        / layers["thermal_conductivity_w_m_k"]
    )

    assemblies = []

    for assembly_name, group in layers.groupby(
        "assembly",
        sort=False,
    ):
        areas = group["area_m2"].dropna()

        if areas.empty:
            raise ValueError(
                f"Для вузла «{assembly_name}» не визначено площу."
            )

        area_min = float(areas.min())
        area_max = float(areas.max())

        # All layers of one plane assembly must cover the same area.
        if abs(area_max - area_min) > 1e-6:
            raise ValueError(
                f"У вузлі «{assembly_name}» шари мають різні площі. "
                "Для послідовної багатошарової стінки площа шарів "
                "має бути однаковою. Якщо це окрема ділянка "
                "(наприклад двері), задайте її як окремий вузол."
            )

        area = float(areas.iloc[0])

        r_layers = float(
            group[
                "layer_resistance_m2_k_w"
            ].sum()
        )

        if r_layers <= 0:
            raise ValueError(
                f"Для вузла «{assembly_name}» сумарний опір шарів "
                "повинен бути більшим за нуль."
            )

        assemblies.append(
            {
                "assembly": assembly_name,
                "area_m2": area,
                "layer_count": int(len(group)),
                "r_layers_m2_k_w": r_layers,
                "u_layers_w_m2_k": 1.0 / r_layers,
            }
        )

    return layers, pd.DataFrame(assemblies)



def air_transport_properties_stage21(
    temperature_c,
    humidity_ratio_kg_kg,
    pressure_kpa,
) -> pd.DataFrame:
    """
    Допоміжні теплофізичні властивості вологого повітря для п.21.

    Густина визначається з уже використаної в моделі психрометрії.
    Динамічна в'язкість сухого повітря — за рівнянням Сазерленда.

    Для діапазону температур сушіння Pr приймається 0.71 як
    стандартне наближення для повітря. Теплопровідність визначається
    з k = mu * cp / Pr, щоб властивості залишались взаємоузгодженими.

    Це саме модель властивостей повітря, а не нові конструктивні
    вихідні дані користувача.
    """
    import numpy as np

    t_c = pd.Series(
        temperature_c,
        dtype="float64",
    )

    w = pd.Series(
        humidity_ratio_kg_kg,
        dtype="float64",
    ).clip(lower=0.0)

    p_kpa = pd.Series(
        pressure_kpa,
        dtype="float64",
    )

    t_k = t_c + 273.15

    # Sutherland equation for dynamic viscosity of air.
    mu0_pa_s = 1.716e-5
    t0_k = 273.15
    s_mu_k = 111.0

    mu_pa_s = (
        mu0_pa_s
        * (t_k / t0_k) ** 1.5
        * (t0_k + s_mu_k)
        / (t_k + s_mu_k)
    )

    rho_da = calculate_dry_air_density_kg_m3(
        temperature_c=t_c,
        humidity_ratio_kg_kg=w,
        pressure_kpa=p_kpa,
    )

    # kg moist air / m3 moist air
    rho_moist = rho_da * (1.0 + w)

    # Heat capacity per kg moist air.
    cp_moist_j_kg_k = (
        1006.0 + 1860.0 * w
    ) / (
        1.0 + w
    )

    pr = pd.Series(
        np.full(len(t_c), 0.71),
        index=t_c.index,
        dtype="float64",
    )

    k_w_m_k = (
        mu_pa_s
        * cp_moist_j_kg_k
        / pr
    )

    return pd.DataFrame(
        {
            "rho_kg_m3": rho_moist,
            "mu_pa_s": mu_pa_s,
            "cp_j_kg_k": cp_moist_j_kg_k,
            "prandtl": pr,
            "k_w_m_k": k_w_m_k,
        },
        index=t_c.index,
    )


def flat_plate_forced_convection_stage21(
    velocity_m_s,
    characteristic_length_m: float,
    temperature_c,
    humidity_ratio_kg_kg,
    pressure_kpa,
) -> pd.DataFrame:
    """
    Середній коефіцієнт примусової конвекції для обтікання
    плоскої поверхні повітрям.

    Re_L = rho * V * L / mu

    Для ламінарного режиму:
        Nu_L = 0.664 * Re_L^0.5 * Pr^(1/3)

    Для Re_L > 5e5 використовується стандартна кореляція
    для турбулентного обтікання пластини з ламінарною
    початковою ділянкою:
        Nu_L = (0.037*Re_L^0.8 - 871) * Pr^(1/3)

    h = Nu_L * k / L

    Якщо V <= 0 або параметри некоректні, h не вигадується,
    а повертається NaN.
    """
    import numpy as np

    if characteristic_length_m <= 0:
        raise ValueError(
            "Характерна довжина для конвекції має бути більшою за нуль."
        )

    v = pd.Series(
        velocity_m_s,
        dtype="float64",
    )

    props = air_transport_properties_stage21(
        temperature_c=temperature_c,
        humidity_ratio_kg_kg=humidity_ratio_kg_kg,
        pressure_kpa=pressure_kpa,
    )

    re_l = (
        props["rho_kg_m3"]
        * v
        * float(characteristic_length_m)
        / props["mu_pa_s"]
    )

    valid = (
        v.notna()
        & (v > 0)
        & re_l.notna()
        & (re_l > 0)
        & props["prandtl"].notna()
        & props["k_w_m_k"].notna()
    )

    nu = pd.Series(
        np.nan,
        index=v.index,
        dtype="float64",
    )

    laminar = valid & (re_l <= 5.0e5)
    turbulent = valid & (re_l > 5.0e5)

    nu.loc[laminar] = (
        0.664
        * re_l.loc[laminar] ** 0.5
        * props.loc[
            laminar,
            "prandtl",
        ] ** (1.0 / 3.0)
    )

    nu_turb = (
        (
            0.037
            * re_l.loc[turbulent] ** 0.8
            - 871.0
        )
        * props.loc[
            turbulent,
            "prandtl",
        ] ** (1.0 / 3.0)
    )

    # Correlation must not generate a negative Nu at transition.
    nu.loc[turbulent] = nu_turb.clip(lower=0.0)

    h = (
        nu
        * props["k_w_m_k"]
        / float(characteristic_length_m)
    )

    regime = pd.Series(
        "не визначено",
        index=v.index,
        dtype="object",
    )

    regime.loc[laminar] = "ламінарний"
    regime.loc[turbulent] = "турбулентний/перехідний"

    return pd.DataFrame(
        {
            "velocity_m_s": v,
            "reynolds": re_l,
            "nusselt": nu,
            "h_w_m2_k": h,
            "flow_regime": regime,
            "rho_kg_m3": props["rho_kg_m3"],
            "mu_pa_s": props["mu_pa_s"],
            "prandtl": props["prandtl"],
            "k_w_m_k": props["k_w_m_k"],
        },
        index=v.index,
    )



def calculate_transmission_heat_losses_stage21(
    coupled_profile: pd.DataFrame,
    zone_profile: pd.DataFrame,
    envelope_assemblies_df: pd.DataFrame,
    geometry: dict,
    step_minutes: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int | str],
]:
    """
    Пункт 21. Автоматичний розрахунок поверхневого теплообміну,
    U(t) та трансмісійних теплових втрат.

    Внутрішня сторона:
      - швидкість біля огородження оцінюється з повної об'ємної
        витрати вентилятора та геометричного перерізу камери;
      - характерна довжина = довжина основного шляху повітря.

    Зовнішня сторона:
      - використовується WS10M із NASA POWER;
      - корекція швидкості вітру з 10 м до висоти сушарки
        НЕ виконується, оскільки для неї потрібні додаткові
        параметри шорсткості/терену, яких у моделі немає;
      - характерна довжина приймається як найбільший зовнішній
        габарит камери. Це геометричне модельне наближення.

    Для кожного вузла:
      R_total,j(t) = 1/h_in(t) + R_layers,j + 1/h_out(t)
      U_j(t) = 1 / R_total,j(t)

    Загалом:
      UA_sum(t) = sum_j U_j(t) * A_j
      Qdot_loss(t) = UA_sum(t) * max(T_ch(t)-T_amb(t), 0)

    Якщо для якогось активного інтервалу неможливо визначити
    h_in або h_out (наприклад, V=0), остаточні втрати цього
    інтервалу не підміняються довільним числом.
    """

    import numpy as np

    required_main = {
        "T2M",
        "PS",
        "WS10M",
        "humidity_ratio_kg_kg",
        "active_drying_command",
        "fan_volume_flow_total_m3_h",
    }

    required_zone = {
        "time_position",
        "air_inlet_temp_c",
        "air_outlet_temp_c",
        "air_inlet_humidity_ratio_g_kg",
        "air_outlet_humidity_ratio_g_kg",
    }

    required_env = {
        "assembly",
        "area_m2",
        "r_layers_m2_k_w",
    }

    missing_main = required_main - set(
        coupled_profile.columns
    )
    missing_zone = required_zone - set(
        zone_profile.columns
    )
    missing_env = required_env - set(
        envelope_assemblies_df.columns
    )

    if missing_main:
        raise ValueError(
            "Для п.21 у часовому профілі відсутні параметри: "
            + ", ".join(sorted(missing_main))
        )

    if missing_zone:
        raise ValueError(
            "Для п.21 у зональному профілі відсутні параметри: "
            + ", ".join(sorted(missing_zone))
        )

    if missing_env:
        raise ValueError(
            "Для п.21 у вузлах огородження відсутні параметри: "
            + ", ".join(sorted(missing_env))
        )

    if step_minutes <= 0:
        raise ValueError(
            "Крок моделювання має бути більшим за нуль."
        )

    gross_flow_area_m2 = float(
        geometry.get(
            "gross_flow_area_m2",
            0.0,
        )
    )

    inside_length_m = float(
        geometry.get(
            "flow_path_length_m",
            0.0,
        )
    )

    chamber_l = float(
        geometry.get(
            "chamber_length_m",
            0.0,
        )
    )
    chamber_w = float(
        geometry.get(
            "chamber_width_m",
            0.0,
        )
    )
    chamber_h = float(
        geometry.get(
            "chamber_height_m",
            0.0,
        )
    )

    outside_length_m = max(
        chamber_l,
        chamber_w,
        chamber_h,
    )

    if gross_flow_area_m2 <= 0:
        raise ValueError(
            "Не визначено додатний геометричний переріз камери."
        )

    if inside_length_m <= 0:
        raise ValueError(
            "Не визначено довжину внутрішнього шляху повітря."
        )

    if outside_length_m <= 0:
        raise ValueError(
            "Не визначено характерний зовнішній розмір камери."
        )

    env = envelope_assemblies_df.copy()

    for col in [
        "area_m2",
        "r_layers_m2_k_w",
    ]:
        env[col] = pd.to_numeric(
            env[col],
            errors="coerce",
        )

    invalid_env = (
        env["area_m2"].isna()
        | (env["area_m2"] <= 0)
        | env["r_layers_m2_k_w"].isna()
        | (env["r_layers_m2_k_w"] <= 0)
    )

    if invalid_env.any():
        bad = env.loc[
            invalid_env,
            "assembly",
        ].astype(str).tolist()

        raise ValueError(
            "Некоректні A або Rшарів для вузлів: "
            + ", ".join(bad)
        )

    # ---------------------------------------------------------
    # Representative internal air state from the zonal model.
    # ---------------------------------------------------------
    zones = zone_profile.copy()

    for col in [
        "air_inlet_temp_c",
        "air_outlet_temp_c",
        "air_inlet_humidity_ratio_g_kg",
        "air_outlet_humidity_ratio_g_kg",
    ]:
        zones[col] = pd.to_numeric(
            zones[col],
            errors="coerce",
        )

    zones["stage21_zone_mean_temp_c"] = (
        zones["air_inlet_temp_c"]
        + zones["air_outlet_temp_c"]
    ) / 2.0

    zones["stage21_zone_mean_w_kg_kg"] = (
        (
            zones["air_inlet_humidity_ratio_g_kg"]
            + zones["air_outlet_humidity_ratio_g_kg"]
        )
        / 2.0
        / 1000.0
    )

    chamber_temp_by_time = (
        zones
        .groupby(
            "time_position",
            sort=True,
        )["stage21_zone_mean_temp_c"]
        .mean()
    )

    chamber_w_by_time = (
        zones
        .groupby(
            "time_position",
            sort=True,
        )["stage21_zone_mean_w_kg_kg"]
        .mean()
    )

    result = coupled_profile.copy()
    result["time_position"] = range(len(result))

    result["heat_loss_chamber_temp_c"] = (
        result["time_position"].map(
            chamber_temp_by_time.to_dict()
        )
    )

    result["heat_loss_chamber_w_kg_kg"] = (
        result["time_position"].map(
            chamber_w_by_time.to_dict()
        )
    )

    result["heat_loss_ambient_temp_c"] = (
        pd.to_numeric(
            result["T2M"],
            errors="coerce",
        )
    )

    result["heat_loss_ambient_w_kg_kg"] = (
        pd.to_numeric(
            result["humidity_ratio_kg_kg"],
            errors="coerce",
        )
    )

    result["heat_loss_pressure_kpa"] = (
        pd.to_numeric(
            result["PS"],
            errors="coerce",
        )
    )

    result["heat_loss_ws10m_m_s"] = (
        pd.to_numeric(
            result["WS10M"],
            errors="coerce",
        )
    )

    fan_flow_m3_h = pd.to_numeric(
        result["fan_volume_flow_total_m3_h"],
        errors="coerce",
    )

    result["heat_loss_inside_bulk_velocity_m_s"] = (
        fan_flow_m3_h
        / 3600.0
        / gross_flow_area_m2
    )

    active = (
        result["active_drying_command"]
        .fillna(False)
        .astype(bool)
    )

    # ---------------------------------------------------------
    # h_in(t)
    # ---------------------------------------------------------
    inside_conv = (
        flat_plate_forced_convection_stage21(
            velocity_m_s=(
                result[
                    "heat_loss_inside_bulk_velocity_m_s"
                ]
            ),
            characteristic_length_m=inside_length_m,
            temperature_c=(
                result["heat_loss_chamber_temp_c"]
            ),
            humidity_ratio_kg_kg=(
                result["heat_loss_chamber_w_kg_kg"]
            ),
            pressure_kpa=(
                result["heat_loss_pressure_kpa"]
            ),
        )
    )

    result["h_inside_w_m2_k"] = (
        inside_conv["h_w_m2_k"]
    ).where(active)

    result["re_inside"] = (
        inside_conv["reynolds"]
    ).where(active)

    result["nu_inside"] = (
        inside_conv["nusselt"]
    ).where(active)

    result["inside_flow_regime"] = (
        inside_conv["flow_regime"]
    ).where(active, "неактивний режим")

    # ---------------------------------------------------------
    # h_out(t), based directly on WS10M.
    # ---------------------------------------------------------
    outside_conv = (
        flat_plate_forced_convection_stage21(
            velocity_m_s=result[
                "heat_loss_ws10m_m_s"
            ],
            characteristic_length_m=outside_length_m,
            temperature_c=(
                result["heat_loss_ambient_temp_c"]
            ),
            humidity_ratio_kg_kg=(
                result["heat_loss_ambient_w_kg_kg"]
            ),
            pressure_kpa=(
                result["heat_loss_pressure_kpa"]
            ),
        )
    )

    result["h_outside_w_m2_k"] = (
        outside_conv["h_w_m2_k"]
    ).where(active)

    result["re_outside"] = (
        outside_conv["reynolds"]
    ).where(active)

    result["nu_outside"] = (
        outside_conv["nusselt"]
    ).where(active)

    result["outside_flow_regime"] = (
        outside_conv["flow_regime"]
    ).where(active, "неактивний режим")

    result["heat_loss_delta_t_c"] = (
        result["heat_loss_chamber_temp_c"]
        - result["heat_loss_ambient_temp_c"]
    )

    # ---------------------------------------------------------
    # Time-dependent U and UA for every assembly.
    # ---------------------------------------------------------
    ua_total = pd.Series(
        0.0,
        index=result.index,
        dtype="float64",
    )

    assembly_records = []

    for _, row_env in env.iterrows():
        assembly_name = str(
            row_env["assembly"]
        )
        area_m2 = float(
            row_env["area_m2"]
        )
        r_layers = float(
            row_env["r_layers_m2_k_w"]
        )

        r_inside = (
            1.0 / result["h_inside_w_m2_k"]
        )

        r_outside = (
            1.0 / result["h_outside_w_m2_k"]
        )

        r_total = (
            r_inside
            + r_layers
            + r_outside
        )

        u_t = 1.0 / r_total
        ua_t = u_t * area_m2

        ua_total = ua_total.add(
            ua_t.fillna(0.0),
            fill_value=0.0,
        )

        valid_u = (
            active
            & u_t.notna()
            & (u_t > 0)
        )

        assembly_records.append(
            {
                "assembly": assembly_name,
                "area_m2": area_m2,
                "r_layers_m2_k_w": r_layers,
                "mean_h_inside_w_m2_k": (
                    float(
                        result.loc[
                            valid_u,
                            "h_inside_w_m2_k",
                        ].mean()
                    )
                    if valid_u.any()
                    else float("nan")
                ),
                "mean_h_outside_w_m2_k": (
                    float(
                        result.loc[
                            valid_u,
                            "h_outside_w_m2_k",
                        ].mean()
                    )
                    if valid_u.any()
                    else float("nan")
                ),
                "mean_r_inside_m2_k_w": (
                    float(
                        (
                            1.0
                            / result.loc[
                                valid_u,
                                "h_inside_w_m2_k",
                            ]
                        ).mean()
                    )
                    if valid_u.any()
                    else float("nan")
                ),
                "mean_r_outside_m2_k_w": (
                    float(
                        (
                            1.0
                            / result.loc[
                                valid_u,
                                "h_outside_w_m2_k",
                            ]
                        ).mean()
                    )
                    if valid_u.any()
                    else float("nan")
                ),
                "mean_u_w_m2_k": (
                    float(
                        u_t.loc[
                            valid_u
                        ].mean()
                    )
                    if valid_u.any()
                    else float("nan")
                ),
                "min_u_w_m2_k": (
                    float(
                        u_t.loc[
                            valid_u
                        ].min()
                    )
                    if valid_u.any()
                    else float("nan")
                ),
                "max_u_w_m2_k": (
                    float(
                        u_t.loc[
                            valid_u
                        ].max()
                    )
                    if valid_u.any()
                    else float("nan")
                ),
            }
        )

    result["heat_loss_ua_total_w_k"] = (
        ua_total.where(active)
    )

    valid_h = (
        active
        & result["h_inside_w_m2_k"].notna()
        & (result["h_inside_w_m2_k"] > 0)
        & result["h_outside_w_m2_k"].notna()
        & (result["h_outside_w_m2_k"] > 0)
    )

    result["heat_loss_calculation_valid"] = (
        valid_h
    )

    positive_delta_t = (
        result["heat_loss_delta_t_c"]
        .clip(lower=0.0)
    )

    result["heat_loss_power_kw"] = (
        result["heat_loss_ua_total_w_k"]
        * positive_delta_t
        / 1000.0
    ).where(
        valid_h,
        float("nan"),
    )

    # Inactive intervals are not part of active drying heat loss.
    result.loc[
        ~active,
        "heat_loss_power_kw",
    ] = 0.0

    result["heat_gain_equivalent_kw"] = (
        result["heat_loss_ua_total_w_k"]
        * (
            -result["heat_loss_delta_t_c"]
        ).clip(lower=0.0)
        / 1000.0
    ).where(
        valid_h,
        float("nan"),
    )

    result.loc[
        ~active,
        "heat_gain_equivalent_kw",
    ] = 0.0

    dt_h = float(step_minutes) / 60.0

    result["heat_loss_energy_kwh_interval"] = (
        result["heat_loss_power_kw"]
        * dt_h
    )

    result["heat_loss_energy_kwh_cumulative"] = (
        result[
            "heat_loss_energy_kwh_interval"
        ]
        .fillna(0.0)
        .cumsum()
    )

    active_count = int(active.sum())
    valid_active_count = int(
        (active & valid_h).sum()
    )

    missing_active_count = (
        active_count - valid_active_count
    )

    if missing_active_count > 0:
        total_heat_loss_kwh = float("nan")
    else:
        total_heat_loss_kwh = float(
            result[
                "heat_loss_energy_kwh_interval"
            ].sum()
        )

    positive_losses = result.loc[
        active
        & valid_h
        & (
            result["heat_loss_power_kw"] > 0
        )
    ]

    peak_heat_loss_kw = (
        float(
            positive_losses[
                "heat_loss_power_kw"
            ].max()
        )
        if not positive_losses.empty
        else 0.0
    )

    mean_heat_loss_kw = (
        float(
            positive_losses[
                "heat_loss_power_kw"
            ].mean()
        )
        if not positive_losses.empty
        else 0.0
    )

    mean_delta_t_c = (
        float(
            positive_losses[
                "heat_loss_delta_t_c"
            ].mean()
        )
        if not positive_losses.empty
        else 0.0
    )

    mean_h_inside = (
        float(
            result.loc[
                active & valid_h,
                "h_inside_w_m2_k",
            ].mean()
        )
        if valid_active_count > 0
        else float("nan")
    )

    mean_h_outside = (
        float(
            result.loc[
                active & valid_h,
                "h_outside_w_m2_k",
            ].mean()
        )
        if valid_active_count > 0
        else float("nan")
    )

    mean_ua = (
        float(
            result.loc[
                active & valid_h,
                "heat_loss_ua_total_w_k",
            ].mean()
        )
        if valid_active_count > 0
        else float("nan")
    )

    total_area = float(
        env["area_m2"].sum()
    )

    mean_u_area_weighted = (
        mean_ua / total_area
        if (
            total_area > 0
            and pd.notna(mean_ua)
        )
        else float("nan")
    )

    summary = {
        "active_interval_count": active_count,
        "valid_active_interval_count": valid_active_count,
        "missing_active_interval_count": missing_active_count,
        "inside_characteristic_length_m": inside_length_m,
        "outside_characteristic_length_m": outside_length_m,
        "gross_flow_area_m2": gross_flow_area_m2,
        "mean_h_inside_w_m2_k": mean_h_inside,
        "mean_h_outside_w_m2_k": mean_h_outside,
        "mean_re_inside": (
            float(
                result.loc[
                    active & valid_h,
                    "re_inside",
                ].mean()
            )
            if valid_active_count > 0
            else float("nan")
        ),
        "mean_re_outside": (
            float(
                result.loc[
                    active & valid_h,
                    "re_outside",
                ].mean()
            )
            if valid_active_count > 0
            else float("nan")
        ),
        "mean_nu_inside": (
            float(
                result.loc[
                    active & valid_h,
                    "nu_inside",
                ].mean()
            )
            if valid_active_count > 0
            else float("nan")
        ),
        "mean_nu_outside": (
            float(
                result.loc[
                    active & valid_h,
                    "nu_outside",
                ].mean()
            )
            if valid_active_count > 0
            else float("nan")
        ),
        "mean_ua_w_k": mean_ua,
        "mean_area_weighted_u_w_m2_k": (
            mean_u_area_weighted
        ),
        "total_transmission_heat_loss_kwh": (
            total_heat_loss_kwh
        ),
        "peak_transmission_heat_loss_kw": (
            peak_heat_loss_kw
        ),
        "mean_transmission_heat_loss_kw": (
            mean_heat_loss_kw
        ),
        "mean_loss_delta_t_c": (
            mean_delta_t_c
        ),
        "external_wind_source": (
            "NASA POWER WS10M, 10 м, без корекції за висотою"
        ),
        "convection_model": (
            "середня примусова конвекція при обтіканні плоскої поверхні"
        ),
    }

    return (
        result,
        pd.DataFrame(assembly_records),
        summary,
    )


def calculate_recuperation_potential_stage22(
    coupled_profile: pd.DataFrame,
    step_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """
    Пункт 22. Теоретичний потенціал рекуперації теплоти
    відпрацьованого сушильного агента.

    На цьому етапі НЕ задається ефективність конкретного
    рекуператора, оскільки його тип і конструкція ще не вибрані.

    Гарячий потік:
        T_h,in = температура повітря після сушильної камери
        w_h    = вологовміст повітря після камери

    Холодний потік:
        T_c,in = температура зовнішнього повітря
        w_c    = зовнішній вологовміст

    Для обох потоків використовується одна і та сама витрата
    сухого повітря, що відповідає поточному повітрообміну сушарки.

    Теплоємнісні витрати:
        C_h = m_dot_da * (1.006 + 1.86*w_h)
        C_c = m_dot_da * (1.006 + 1.86*w_c)

    Теоретично максимальна потужність відчутної рекуперації:
        Qdot_max = C_min * max(T_h,in - T_c,in, 0)

    Для довільної ефективності epsilon:
        Qdot_rec = epsilon * Qdot_max

        T_c,out = T_c,in + Qdot_rec / C_c
        T_h,out = T_h,in - Qdot_rec / C_h

    Оскільки epsilon конкретного апарата ще невідома, програма
    формує параметричну таблицю для epsilon=0...1, а не підставляє
    довільне проєктне значення.

    Додатково виконується лише діагностична перевірка:
        T_h,out <= T_dp,h
    тобто чи опустилась би середня температура гарячого потоку
    нижче його точки роси. Це НЕ гарантує відсутності конденсації
    на поверхні теплообмінника, бо для цього потрібна локальна
    температура стінки та геометрія рекуператора.
    """
    import numpy as np

    required = {
        "T2M",
        "PS",
        "humidity_ratio_kg_kg",
        "drying_air_temperature_c",
        "dry_air_mass_flow_total_kg_h",
        "active_drying_command",
        "zonal_air_outlet_temp_c",
        "zonal_air_outlet_humidity_ratio_kg_kg",
    }

    missing = required - set(coupled_profile.columns)
    if missing:
        raise ValueError(
            "Для п.22 відсутні параметри: "
            + ", ".join(sorted(missing))
        )

    if step_minutes <= 0:
        raise ValueError(
            "Крок моделювання має бути більшим за нуль."
        )

    result = coupled_profile.copy()

    t_hot = pd.to_numeric(
        result["zonal_air_outlet_temp_c"],
        errors="coerce",
    )

    w_hot = pd.to_numeric(
        result["zonal_air_outlet_humidity_ratio_kg_kg"],
        errors="coerce",
    )

    t_cold = pd.to_numeric(
        result["T2M"],
        errors="coerce",
    )

    w_cold = pd.to_numeric(
        result["humidity_ratio_kg_kg"],
        errors="coerce",
    )

    t_target = pd.to_numeric(
        result["drying_air_temperature_c"],
        errors="coerce",
    )

    pressure_kpa = pd.to_numeric(
        result["PS"],
        errors="coerce",
    )

    mdot_da_kg_h = pd.to_numeric(
        result["dry_air_mass_flow_total_kg_h"],
        errors="coerce",
    )

    mdot_da_kg_s = (
        mdot_da_kg_h / 3600.0
    )

    active = (
        result["active_drying_command"]
        .fillna(False)
        .astype(bool)
    )

    cp_hot_kj_kg_da_k = (
        1.006 + 1.86 * w_hot
    )

    cp_cold_kj_kg_da_k = (
        1.006 + 1.86 * w_cold
    )

    c_hot_kw_k = (
        mdot_da_kg_s
        * cp_hot_kj_kg_da_k
    )

    c_cold_kw_k = (
        mdot_da_kg_s
        * cp_cold_kj_kg_da_k
    )

    c_min_kw_k = pd.concat(
        [
            c_hot_kw_k.rename("hot"),
            c_cold_kw_k.rename("cold"),
        ],
        axis=1,
    ).min(axis=1)

    c_max_kw_k = pd.concat(
        [
            c_hot_kw_k.rename("hot"),
            c_cold_kw_k.rename("cold"),
        ],
        axis=1,
    ).max(axis=1)

    capacity_ratio = (
        c_min_kw_k / c_max_kw_k
    ).where(
        c_max_kw_k > 0
    )

    delta_t_in_c = (
        t_hot - t_cold
    )

    qmax_kw = (
        c_min_kw_k
        * delta_t_in_c.clip(lower=0.0)
    ).where(
        active,
        0.0,
    )

    # Baseline sensible heating demand of fresh air to target.
    q_baseline_kw = (
        c_cold_kw_k
        * (t_target - t_cold).clip(lower=0.0)
    ).where(
        active,
        0.0,
    )

    # Hot-stream dew point from humidity ratio and total pressure.
    p_v_hot_kpa = (
        w_hot
        * pressure_kpa
        / (0.621945 + w_hot)
    )

    gamma_hot = np.log(
        p_v_hot_kpa / 0.61094
    )

    t_dp_hot_c = (
        243.04
        * gamma_hot
        / (17.625 - gamma_hot)
    )

    # Effectiveness at which the bulk hot outlet would just reach
    # the inlet hot-stream dew point.
    q_bulk_dew_limit_kw = (
        c_hot_kw_k
        * (t_hot - t_dp_hot_c).clip(lower=0.0)
    )

    epsilon_bulk_dew_limit = (
        q_bulk_dew_limit_kw / qmax_kw
    ).where(
        qmax_kw > 0
    )

    epsilon_bulk_dew_limit = (
        epsilon_bulk_dew_limit
        .clip(lower=0.0, upper=1.0)
    )

    dt_h = float(step_minutes) / 60.0

    result["recup_hot_in_temp_c"] = t_hot
    result["recup_hot_in_w_kg_kg"] = w_hot
    result["recup_hot_in_dew_point_c"] = t_dp_hot_c

    result["recup_cold_in_temp_c"] = t_cold
    result["recup_cold_in_w_kg_kg"] = w_cold

    result["recup_c_hot_kw_k"] = c_hot_kw_k
    result["recup_c_cold_kw_k"] = c_cold_kw_k
    result["recup_c_min_kw_k"] = c_min_kw_k
    result["recup_capacity_ratio"] = capacity_ratio

    result["recup_delta_t_in_c"] = delta_t_in_c
    result["recup_qmax_sensible_kw"] = qmax_kw
    result["recup_baseline_air_heating_kw"] = q_baseline_kw

    result["recup_bulk_dew_epsilon_limit"] = (
        epsilon_bulk_dew_limit
    )

    result["recup_qmax_energy_kwh_interval"] = (
        qmax_kw * dt_h
    )

    result["recup_qmax_energy_kwh_cumulative"] = (
        result[
            "recup_qmax_energy_kwh_interval"
        ].cumsum()
    )

    # ---------------------------------------------------------
    # Parametric effectiveness table. No arbitrary design value.
    # ---------------------------------------------------------
    epsilon_values = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    scenario_rows = []

    active_recovery = (
        active
        & qmax_kw.notna()
        & (qmax_kw > 0)
        & (c_hot_kw_k > 0)
        & (c_cold_kw_k > 0)
    )

    baseline_energy_kwh = float(
        (q_baseline_kw * dt_h).sum()
    )

    qmax_energy_kwh = float(
        (
            result[
                "recup_qmax_energy_kwh_interval"
            ]
        ).sum()
    )

    for epsilon in epsilon_values:
        q_rec_kw = (
            float(epsilon) * qmax_kw
        )

        t_cold_out = (
            t_cold
            + q_rec_kw / c_cold_kw_k
        )

        t_hot_out = (
            t_hot
            - q_rec_kw / c_hot_kw_k
        )

        residual_heating_kw = (
            c_cold_kw_k
            * (
                t_target
                - t_cold_out
            ).clip(lower=0.0)
        ).where(
            active,
            0.0,
        )

        condensation_diag = (
            active_recovery
            & t_hot_out.notna()
            & t_dp_hot_c.notna()
            & (
                t_hot_out
                <= t_dp_hot_c
            )
        )

        recovered_energy_kwh = float(
            (q_rec_kw * dt_h).sum()
        )

        residual_energy_kwh = float(
            (
                residual_heating_kw
                * dt_h
            ).sum()
        )

        scenario_rows.append(
            {
                "effectiveness": float(epsilon),
                "recovered_energy_kwh": (
                    recovered_energy_kwh
                ),
                "residual_air_heating_kwh": (
                    residual_energy_kwh
                ),
                "air_heating_reduction_pct": (
                    (
                        1.0
                        - residual_energy_kwh
                        / baseline_energy_kwh
                    )
                    * 100.0
                    if baseline_energy_kwh > 0
                    else float("nan")
                ),
                "mean_cold_outlet_temp_c": (
                    float(
                        t_cold_out.loc[
                            active_recovery
                        ].mean()
                    )
                    if active_recovery.any()
                    else float("nan")
                ),
                "mean_hot_outlet_temp_c": (
                    float(
                        t_hot_out.loc[
                            active_recovery
                        ].mean()
                    )
                    if active_recovery.any()
                    else float("nan")
                ),
                "bulk_dewpoint_risk_intervals": int(
                    condensation_diag.sum()
                ),
            }
        )

    scenarios = pd.DataFrame(
        scenario_rows
    )

    valid_eps_limit = (
        epsilon_bulk_dew_limit.loc[
            active_recovery
        ]
        .dropna()
    )

    minimum_bulk_dew_limit = (
        float(valid_eps_limit.min())
        if not valid_eps_limit.empty
        else float("nan")
    )

    mean_hot_in_temp = (
        float(
            t_hot.loc[
                active_recovery
            ].mean()
        )
        if active_recovery.any()
        else float("nan")
    )

    mean_cold_in_temp = (
        float(
            t_cold.loc[
                active_recovery
            ].mean()
        )
        if active_recovery.any()
        else float("nan")
    )

    mean_hot_dp = (
        float(
            t_dp_hot_c.loc[
                active_recovery
            ].mean()
        )
        if active_recovery.any()
        else float("nan")
    )

    peak_qmax_kw = (
        float(
            qmax_kw.loc[
                active_recovery
            ].max()
        )
        if active_recovery.any()
        else 0.0
    )

    mean_qmax_kw = (
        float(
            qmax_kw.loc[
                active_recovery
            ].mean()
        )
        if active_recovery.any()
        else 0.0
    )

    summary = {
        "active_recovery_interval_count": int(
            active_recovery.sum()
        ),
        "mean_hot_inlet_temp_c": (
            mean_hot_in_temp
        ),
        "mean_cold_inlet_temp_c": (
            mean_cold_in_temp
        ),
        "mean_hot_inlet_dew_point_c": (
            mean_hot_dp
        ),
        "peak_qmax_sensible_kw": (
            peak_qmax_kw
        ),
        "mean_qmax_sensible_kw": (
            mean_qmax_kw
        ),
        "qmax_sensible_energy_kwh": (
            qmax_energy_kwh
        ),
        "baseline_air_heating_energy_kwh": (
            baseline_energy_kwh
        ),
        "maximum_theoretical_heating_offset_pct": (
            (
                qmax_energy_kwh
                / baseline_energy_kwh
                * 100.0
            )
            if baseline_energy_kwh > 0
            else float("nan")
        ),
        "minimum_bulk_dewpoint_effectiveness_limit": (
            minimum_bulk_dew_limit
        ),
        "model_type": (
            "сухий відчутний повітря-повітря рекуператор; "
            "параметричний аналіз ефективності"
        ),
    }

    return result, scenarios, summary




def calculate_unified_heat_balance_stage23(
    air_heating_profile: pd.DataFrame,
    heat_loss_profile: pd.DataFrame,
    recuperation_profile: pd.DataFrame,
    product_summary: dict | None,
    evaporation_summary: dict | None,
    construction_summary: dict | None,
    step_minutes: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int | str],
]:
    """
    Пункт 23. Зведений нестаціонарний тепловий баланс.

    Мета цього етапу — об'єднати ТІЛЬКИ ті складові, які вже
    розраховані без подвійного обліку.

    Динамічно визначені складові:
        Qdot_air(t)   — нагрівання свіжого сушильного агента, п.17;
        Qdot_rec(t)   — повернена відчутна теплота, п.22;
        Qdot_loss(t)  — трансмісійні втрати через огородження, п.21.

    Для заданого сценарію ефективності рекуператора epsilon:
        Qdot_air,after_rec(t) =
            max[Qdot_air(t) - epsilon*Qdot_max_rec(t), 0]

        Qdot_known(t) =
            Qdot_air,after_rec(t) + Qdot_loss(t)

    ВАЖЛИВО ПРО ВИПАРОВУВАННЯ:
    у зональній моделі п.15 повітря проходить через зерно
    приблизно по лінії сталої ентальпії:
        h_air,out ≈ h_air,in
    а додана водяна пара супроводжується охолодженням повітря.
    Тому Q_evap з п.18 є діагностичною енергетичною
    характеристикою фазового переходу і НЕ додається вдруге
    до зовнішнього теплового навантаження.

    НАГРІВАННЯ ПРОДУКТУ:
    п.19 поки дає лише інтегральну проєктну оцінку Q_product,
    але не Qdot_product(t). Тому її можна показати в балансі
    циклу, але не можна довільно розподіляти по часових кроках.

    КОНСТРУКЦІЇ:
    п.20 дає C_constr = sum(m_i*c_p,i), але без T_constr(t)
    неможливо визначити:
        Qdot_constr(t) = C_constr * dT_constr/dt
    Тому теплота накопичення конструкцією тут НЕ вигадується
    і залишається незамкненим членом балансу.

    Таким чином цей етап формує:
      1) коректний динамічний баланс уже визначених складових;
      2) енергетичний реєстр складових, які не слід додавати двічі;
      3) перелік ще невизначених членів до повного замикання балансу.
    """
    import numpy as np

    if step_minutes <= 0:
        raise ValueError(
            "Крок моделювання має бути більшим за нуль."
        )

    n_air = len(air_heating_profile)
    n_loss = len(heat_loss_profile)
    n_rec = len(recuperation_profile)

    if not (
        n_air == n_loss == n_rec
    ):
        raise ValueError(
            "Часові профілі п.17, п.21 і п.22 мають різну довжину."
        )

    required_air = {
        "air_heating_power_kw",
        "active_drying_command",
        "dry_air_mass_flow_total_kg_h",
        "zonal_air_inlet_temp_c",
        "zonal_air_inlet_humidity_ratio_kg_kg",
        "zonal_air_outlet_temp_c",
        "zonal_air_outlet_humidity_ratio_kg_kg",
    }

    required_loss = {
        "heat_loss_power_kw",
    }

    required_rec = {
        "recup_qmax_sensible_kw",
        "recup_baseline_air_heating_kw",
    }

    missing_air = required_air - set(
        air_heating_profile.columns
    )
    missing_loss = required_loss - set(
        heat_loss_profile.columns
    )
    missing_rec = required_rec - set(
        recuperation_profile.columns
    )

    if missing_air:
        raise ValueError(
            "Для п.23 у профілі п.17 відсутні: "
            + ", ".join(sorted(missing_air))
        )

    if missing_loss:
        raise ValueError(
            "Для п.23 у профілі п.21 відсутні: "
            + ", ".join(sorted(missing_loss))
        )

    if missing_rec:
        raise ValueError(
            "Для п.23 у профілі п.22 відсутні: "
            + ", ".join(sorted(missing_rec))
        )

    result = air_heating_profile.copy()

    q_air_kw = pd.to_numeric(
        air_heating_profile[
            "air_heating_power_kw"
        ],
        errors="coerce",
    ).reset_index(drop=True)

    q_loss_kw = pd.to_numeric(
        heat_loss_profile[
            "heat_loss_power_kw"
        ],
        errors="coerce",
    ).reset_index(drop=True)

    q_rec_max_kw = pd.to_numeric(
        recuperation_profile[
            "recup_qmax_sensible_kw"
        ],
        errors="coerce",
    ).reset_index(drop=True)

    q_rec_baseline_kw = pd.to_numeric(
        recuperation_profile[
            "recup_baseline_air_heating_kw"
        ],
        errors="coerce",
    ).reset_index(drop=True)

    active = (
        air_heating_profile[
            "active_drying_command"
        ]
        .fillna(False)
        .astype(bool)
        .reset_index(drop=True)
    )

    # A missing transmission-loss result in an active interval means
    # the dynamic balance is not closed for that interval.
    invalid_loss = (
        active
        & q_loss_kw.isna()
    )

    if invalid_loss.any():
        raise ValueError(
            "П.21 не має валідного значення теплових втрат "
            f"для {int(invalid_loss.sum())} активних інтервалів."
        )

    q_loss_kw = q_loss_kw.fillna(0.0)

    q_air_kw = q_air_kw.fillna(0.0)
    q_rec_max_kw = q_rec_max_kw.fillna(0.0)
    q_rec_baseline_kw = (
        q_rec_baseline_kw.fillna(0.0)
    )

    dt_h = float(step_minutes) / 60.0

    # ---------------------------------------------------------
    # Cross-check 1:
    # Stage-22 baseline fresh-air heating should reproduce stage 17.
    # ---------------------------------------------------------
    baseline_check_kw = (
        q_air_kw
        - q_rec_baseline_kw
    )

    max_baseline_check_error_kw = float(
        baseline_check_kw.abs().max()
    )

    # ---------------------------------------------------------
    # Cross-check 2:
    # Stage-15 chamber air is treated along constant enthalpy.
    # This is the key reason why stage-18 evaporation energy must
    # not be added again as an external heat-demand term.
    # ---------------------------------------------------------
    t_in = pd.to_numeric(
        air_heating_profile[
            "zonal_air_inlet_temp_c"
        ],
        errors="coerce",
    ).reset_index(drop=True)

    w_in = pd.to_numeric(
        air_heating_profile[
            "zonal_air_inlet_humidity_ratio_kg_kg"
        ],
        errors="coerce",
    ).reset_index(drop=True)

    t_out = pd.to_numeric(
        air_heating_profile[
            "zonal_air_outlet_temp_c"
        ],
        errors="coerce",
    ).reset_index(drop=True)

    w_out = pd.to_numeric(
        air_heating_profile[
            "zonal_air_outlet_humidity_ratio_kg_kg"
        ],
        errors="coerce",
    ).reset_index(drop=True)

    mdot_da_kg_s = (
        pd.to_numeric(
            air_heating_profile[
                "dry_air_mass_flow_total_kg_h"
            ],
            errors="coerce",
        )
        .reset_index(drop=True)
        / 3600.0
    )

    h_in = moist_air_enthalpy_kj_kg_da(
        temperature_c=t_in,
        humidity_ratio_kg_kg=w_in,
    )
    h_in.index = result.index

    h_out = moist_air_enthalpy_kj_kg_da(
        temperature_c=t_out,
        humidity_ratio_kg_kg=w_out,
    )
    h_out.index = result.index

    chamber_air_enthalpy_delta_kw = (
        mdot_da_kg_s.to_numpy()
        * (
            h_out.to_numpy()
            - h_in.to_numpy()
        )
    )

    chamber_air_enthalpy_delta_kw = pd.Series(
        chamber_air_enthalpy_delta_kw,
        index=result.index,
        dtype="float64",
    )

    chamber_air_enthalpy_delta_kw = (
        chamber_air_enthalpy_delta_kw.where(
            active.to_numpy(),
            0.0,
        )
    )

    max_chamber_enthalpy_balance_error_kw = float(
        chamber_air_enthalpy_delta_kw
        .abs()
        .max()
    )

    # Store base diagnostic columns.
    result["balance_air_heating_no_rec_kw"] = (
        q_air_kw.to_numpy()
    )

    result["balance_transmission_loss_kw"] = (
        q_loss_kw.to_numpy()
    )

    result["balance_recuperation_qmax_kw"] = (
        q_rec_max_kw.to_numpy()
    )

    result[
        "balance_stage17_vs_stage22_check_kw"
    ] = baseline_check_kw.to_numpy()

    result[
        "balance_chamber_air_enthalpy_delta_kw"
    ] = chamber_air_enthalpy_delta_kw.to_numpy()

    # ---------------------------------------------------------
    # Parametric balance versus recuperator effectiveness.
    # No arbitrary design epsilon is selected.
    # ---------------------------------------------------------
    epsilon_values = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    product_design_kwh = (
        float(
            product_summary[
                "total_initial_product_sensible_heat_kwh"
            ]
        )
        if (
            product_summary is not None
            and "total_initial_product_sensible_heat_kwh"
            in product_summary
        )
        else float("nan")
    )

    evaporation_diag_kwh = (
        float(
            evaporation_summary[
                "total_evaporation_energy_kwh"
            ]
        )
        if (
            evaporation_summary is not None
            and "total_evaporation_energy_kwh"
            in evaporation_summary
        )
        else float("nan")
    )

    construction_capacity_kj_k = (
        float(
            construction_summary[
                "total_thermal_capacity_kj_k"
            ]
        )
        if (
            construction_summary is not None
            and "total_thermal_capacity_kj_k"
            in construction_summary
        )
        else float("nan")
    )

    q_air_no_rec_energy_kwh = float(
        (q_air_kw * dt_h).sum()
    )

    q_loss_energy_kwh = float(
        (q_loss_kw * dt_h).sum()
    )

    scenario_rows = []

    profile_eps_columns = {}

    for epsilon in epsilon_values:
        q_rec_kw = (
            float(epsilon)
            * q_rec_max_kw
        )

        # Recovery cannot reduce the external fresh-air heating below zero.
        q_air_after_rec_kw = (
            q_air_kw - q_rec_kw
        ).clip(lower=0.0)

        q_known_kw = (
            q_air_after_rec_kw
            + q_loss_kw
        )

        e_rec_kwh = float(
            (q_rec_kw * dt_h).sum()
        )

        e_air_after_rec_kwh = float(
            (
                q_air_after_rec_kw
                * dt_h
            ).sum()
        )

        e_known_dynamic_kwh = float(
            (
                q_known_kw
                * dt_h
            ).sum()
        )

        e_with_product_design_kwh = (
            e_known_dynamic_kwh
            + product_design_kwh
            if pd.notna(product_design_kwh)
            else float("nan")
        )

        scenario_rows.append(
            {
                "effectiveness": float(epsilon),
                "recovered_sensible_energy_kwh": (
                    e_rec_kwh
                ),
                "fresh_air_heating_after_rec_kwh": (
                    e_air_after_rec_kwh
                ),
                "transmission_loss_kwh": (
                    q_loss_energy_kwh
                ),
                "known_dynamic_source_energy_kwh": (
                    e_known_dynamic_kwh
                ),
                "product_design_heat_kwh": (
                    product_design_kwh
                ),
                "known_cycle_energy_plus_product_kwh": (
                    e_with_product_design_kwh
                ),
                "construction_storage_status": (
                    "не визначено без Tконстр(t)"
                ),
                "source_reduction_vs_no_rec_pct": (
                    (
                        1.0
                        - e_known_dynamic_kwh
                        / (
                            q_air_no_rec_energy_kwh
                            + q_loss_energy_kwh
                        )
                    )
                    * 100.0
                    if (
                        q_air_no_rec_energy_kwh
                        + q_loss_energy_kwh
                    ) > 0
                    else float("nan")
                ),
            }
        )

        # Keep three illustrative time profiles only.
        if epsilon in {
            0.0,
            0.5,
            1.0,
        }:
            profile_eps_columns[
                f"balance_known_power_eps_{int(epsilon*100)}_kw"
            ] = q_known_kw.to_numpy()

    for col, values in profile_eps_columns.items():
        result[col] = values

    scenarios = pd.DataFrame(
        scenario_rows
    )

    # ---------------------------------------------------------
    # Energy ledger: explicitly states what is added and what is not.
    # ---------------------------------------------------------
    ledger_rows = [
        {
            "component": (
                "Нагрівання свіжого сушильного агента, п.17"
            ),
            "role": (
                "Додається до зовнішньої теплової потреби "
                "до врахування рекуперації"
            ),
            "energy_kwh": (
                q_air_no_rec_energy_kwh
            ),
            "time_resolved": "так",
        },
        {
            "component": (
                "Рекуперація відпрацьованого повітря, п.22"
            ),
            "role": (
                "Віднімається; залежить від εрек"
            ),
            "energy_kwh": float("nan"),
            "time_resolved": "так",
        },
        {
            "component": (
                "Трансмісійні втрати через огородження, п.21"
            ),
            "role": (
                "Додаються до зовнішньої теплової потреби"
            ),
            "energy_kwh": (
                q_loss_energy_kwh
            ),
            "time_resolved": "так",
        },
        {
            "component": (
                "Теплота фазового переходу води, п.18"
            ),
            "role": (
                "НЕ додається окремо: використовується як "
                "діагностична характеристика, бо зональна модель "
                "п.15 уже переносить її через охолодження/зволоження "
                "повітря при майже сталій ентальпії"
            ),
            "energy_kwh": (
                evaporation_diag_kwh
            ),
            "time_resolved": "діагностично",
        },
        {
            "component": (
                "Нагрівання продукту, п.19"
            ),
            "role": (
                "Може бути додано лише на рівні енергії циклу; "
                "часовий профіль Q̇прод(t) ще не визначено"
            ),
            "energy_kwh": (
                product_design_kwh
            ),
            "time_resolved": "ні",
        },
        {
            "component": (
                "Акумулювання теплоти конструкцією, п.20"
            ),
            "role": (
                "Ще не додається: відоме Cконстр, але невідоме "
                "Tконстр(t)"
            ),
            "energy_kwh": float("nan"),
            "time_resolved": "ні",
        },
    ]

    ledger = pd.DataFrame(
        ledger_rows
    )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------
    no_rec_dynamic_kwh = (
        q_air_no_rec_energy_kwh
        + q_loss_energy_kwh
    )

    eps50_row = scenarios.loc[
        scenarios["effectiveness"] == 0.5
    ].iloc[0]

    eps100_row = scenarios.loc[
        scenarios["effectiveness"] == 1.0
    ].iloc[0]

    summary = {
        "air_heating_no_rec_kwh": (
            q_air_no_rec_energy_kwh
        ),
        "transmission_loss_kwh": (
            q_loss_energy_kwh
        ),
        "known_dynamic_no_rec_kwh": (
            no_rec_dynamic_kwh
        ),
        "product_design_heat_kwh": (
            product_design_kwh
        ),
        "evaporation_diagnostic_kwh": (
            evaporation_diag_kwh
        ),
        "construction_capacity_kj_k": (
            construction_capacity_kj_k
        ),
        "known_dynamic_eps50_kwh": float(
            eps50_row[
                "known_dynamic_source_energy_kwh"
            ]
        ),
        "known_dynamic_eps100_kwh": float(
            eps100_row[
                "known_dynamic_source_energy_kwh"
            ]
        ),
        "max_stage17_stage22_baseline_error_kw": (
            max_baseline_check_error_kw
        ),
        "max_chamber_air_enthalpy_balance_error_kw": (
            max_chamber_enthalpy_balance_error_kw
        ),
        "balance_closure_status": (
            "частково замкнений: динамічно замкнені повітря, "
            "рекуперація та трансмісійні втрати; "
            "Q̇прод(t) і Q̇констр(t) ще не визначені"
        ),
    }

    return result, scenarios, ledger, summary



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
    "Етапи 1–23: вихідні дані, масовий баланс, режими, "
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

tab_product, tab_process, tab_weather, tab_psychro, tab_removal, tab_geometry, tab_kinetics, tab_equilibrium, tab_heat_mass, tab_coupled, tab_performance, tab_air_heating, tab_evaporation, tab_product_heating, tab_construction_heating, tab_heat_losses, tab_recuperation, tab_heat_balance = st.tabs(
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
        "10. Спільний розрахунок сушіння",
        "11. Перевірка продуктивності",
        "12. Нагрівання сушильного агента",
        "13. Випаровування вологи",
        "14. Нагрівання продукту",
        "15. Конструкція і теплоємність",
        "16. Теплові втрати",
        "17. Рекуперація",
        "18. Зведений тепловий баланс",
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
        "Локальна швидкість через шар v_Page, м/с",
        min_value=0.15,
        max_value=1.00,
        value=float(default_velocity),
        step=0.05,
        format="%.2f",
        key="stage14_air_velocity_m_s",
    )

    st.caption(
        "Це локальна швидкість сушильного агента, яка входить "
        "у кореляцію Page. У Ramaj et al. повітря проходило через "
        "поровий об'єм зерна в циліндричному тримачі з перфорованим "
        "дном. Тому цю швидкість не слід автоматично множити на "
        "повний поперечний переріз сушильної камери."
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

# ---------------------------------------------------------------------
# 10. СПІЛЬНИЙ РОЗРАХУНОК СУШІННЯ
# ---------------------------------------------------------------------
with tab_coupled:
    st.subheader(
        "Етап 15. Зональний спільний розрахунок сушіння"
    )

    st.info(
        "На цьому етапі локальна швидкість v_Page відділена від "
        "загальної продуктивності вентилятора. У Ramaj et al. "
        "швидкість вимірювалася для повітря, що проходило через "
        "поровий об'єм зерна у тримачі з перфорованим дном. "
        "Тому V̇вент визначається через активну площу проходу "
        "безпосередньо через продукт, а не через повний переріз камери."
    )

    st.markdown("#### 1. Обмеження вихідної Page-моделі")

    source_x0 = 0.159
    source_x0_unc = 0.001
    source_w0 = source_x0 / (1.0 + source_x0) * 100.0

    model_x0 = (
        float(product["initial_moisture_pct"]) / 100.0
    ) / (
        1.0
        - float(product["initial_moisture_pct"]) / 100.0
    )

    st.write(
        "Ramaj et al. (2021): "
        f"**X₀={source_x0:.3f}±{source_x0_unc:.3f} кг/кг d.b.** "
        f"(≈ **{source_w0:.2f} % w.b.**). "
        "Наша модель: "
        f"**X₀={model_x0:.3f} кг/кг d.b.** "
        f"({float(product['initial_moisture_pct']):.1f} % w.b.)."
    )

    if model_x0 > source_x0 + source_x0_unc:
        st.warning(
            "Початкова частина процесу є екстраполяцією Page "
            "за станом продукту. Вона маркується окремо; "
            "штучний коригувальний коефіцієнт не вводиться."
        )

    if "heat_mass_transfer_input_profile" not in st.session_state:
        st.warning(
            "Спочатку виконайте етап 14 у вкладці "
            "«Тепло- і масообмін»."
        )
    elif "dryer_geometry" not in st.session_state:
        st.warning(
            "Спочатку задайте геометрію сушильної камери."
        )
    else:
        stage15_input = st.session_state[
            "heat_mass_transfer_input_profile"
        ]

        geometry15 = st.session_state[
            "dryer_geometry"
        ]

        tray_area = float(
            geometry15["tray_area_m2"]
        )
        tray_count_geom = int(
            geometry15.get("tray_count", 1)
        )
        total_drying_area = float(
            geometry15["total_drying_area_m2"]
        )
        chamber_cross_section = float(
            geometry15["gross_flow_area_m2"]
        )

        st.markdown(
            "#### 2. Схема повітророзподілу через продукт"
        )

        airflow_scheme = st.radio(
            "Оберіть фізичну схему проходження повітря",
            options=[
                "Послідовний тракт через лотки",
                "Паралельний розподіл між лотками",
                "Користувацька кількість паралельних трактів",
            ],
            index=0,
            key="stage15_airflow_scheme",
        )

        if airflow_scheme == "Послідовний тракт через лотки":
            parallel_path_count = 1
            st.write(
                "Одне й те саме повітря послідовно проходить "
                "через зони/шари продукту. Для зв'язку з v_Page "
                "використовується площа одного активного шару."
            )

        elif airflow_scheme == "Паралельний розподіл між лотками":
            parallel_path_count = tray_count_geom
            st.write(
                "Загальний потік вентилятора ділиться між "
                f"**{tray_count_geom}** однаковими паралельними "
                "трактами. Кожен тракт має ту саму локальну "
                "швидкість v_Page."
            )

        else:
            parallel_path_count = int(
                st.number_input(
                    "Кількість однакових паралельних трактів",
                    min_value=1,
                    max_value=50,
                    value=max(1, tray_count_geom),
                    step=1,
                    key="stage15_parallel_path_count",
                )
            )

        st.markdown(
            "#### 3. Активна площа одного повітряного тракту"
        )

        st.write(
            f"Корисна площа одного лотка за геометрією: "
            f"**{tray_area:.4f} м²**. "
            f"Сумарна площа сушіння: **{total_drying_area:.4f} м²**. "
            f"Повний переріз камери: **{chamber_cross_section:.4f} м²**."
        )

        area_basis = st.radio(
            "Як задати площу, перпендикулярну локальному потоку через продукт?",
            options=[
                "Площа одного лотка",
                "Задати вручну",
            ],
            index=0,
            key="stage15_area_basis",
        )

        if area_basis == "Площа одного лотка":
            active_area_per_path = tray_area
        else:
            active_area_per_path = st.number_input(
                "Активна площа одного тракту Aшар, м²",
                min_value=0.001,
                value=float(tray_area),
                step=0.01,
                format="%.4f",
                key="stage15_active_area_per_path_m2",
            )

        st.caption(
            "Це площа поперечного перерізу шару, до якої відноситься "
            "локальна superficial velocity v_Page. Вона не є сумою "
            "площ отворів перфорації. Для packed/thin-layer потоку "
            "superficial velocity відноситься до повної площі "
            "поперечного перерізу шару."
        )

        st.markdown(
            "#### 4. Зв'язок v_Page із продуктивністю вентилятора"
        )

        page_velocity_series = pd.to_numeric(
            stage15_input["air_velocity_m_s"],
            errors="coerce",
        )

        v_page_design = float(
            page_velocity_series.dropna().iloc[0]
            if not page_velocity_series.dropna().empty
            else 0.50
        )

        flow_one_path_m3_h = (
            v_page_design
            * float(active_area_per_path)
            * 3600.0
        )

        flow_total_m3_h = (
            flow_one_path_m3_h
            * int(parallel_path_count)
        )

        old_wrong_mapping_m3_h = (
            v_page_design
            * chamber_cross_section
            * 3600.0
        )

        st.latex(
            r"\dot V_{\mathrm{тракт}}"
            r"=v_{\mathrm{Page}}A_{\mathrm{шар}}"
        )

        st.latex(
            r"\dot V_{\mathrm{вент}}"
            r"=N_{\parallel}\dot V_{\mathrm{тракт}}"
        )

        f1, f2, f3, f4 = st.columns(4)

        f1.metric(
            "v_Page",
            f"{v_page_design:.2f} м/с",
        )

        f2.metric(
            "A шару одного тракту",
            f"{float(active_area_per_path):.3f} м²",
        )

        f3.metric(
            "Витрата одного тракту",
            f"{flow_one_path_m3_h:.0f} м³/год",
        )

        f4.metric(
            "Загальна витрата вентилятора",
            f"{flow_total_m3_h:.0f} м³/год",
        )

        st.write(
            "Для порівняння, стара схема автоматично множила "
            f"v_Page на повний переріз камери і давала "
            f"**{old_wrong_mapping_m3_h:.0f} м³/год**. "
            "Це значення більше не використовується у розрахунку."
        )

        if (
            airflow_scheme
            == "Паралельний розподіл між лотками"
        ):
            st.info(
                "У паралельній схемі зональна модель розраховує "
                "один репрезентативний тракт із 1/N частиною "
                "продукту, а потім масштабує видалену воду та "
                "витрату повітря на всі однакові тракти."
            )

        st.markdown(
            "#### 5. Дискретизація репрезентативного тракту"
        )

        zone_count_stage15 = st.number_input(
            "Кількість розрахункових зон уздовж одного тракту",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            key="zone_count_stage15",
        )

        st.caption(
            "Це чисельна дискретизація вздовж потоку, а не "
            "кількість лотків."
        )

        current_balance15 = calculate_mass_balance(
            float(initial_mass_kg),
            float(product["initial_moisture_pct"]),
            float(product["final_moisture_pct"]),
        )

        try:
            (
                coupled_profile,
                coupled_zone_profile,
                coupled_summary,
            ) = simulate_zonal_coupled_drying_stage15(
                input_profile=stage15_input,
                dry_matter_kg=float(
                    current_balance15["dry_matter_kg"]
                ),
                initial_x_db=float(
                    current_balance15["initial_dry_basis"]
                ),
                target_x_db=float(
                    current_balance15["final_dry_basis"]
                ),
                step_minutes=int(model_step_minutes),
                active_flow_area_per_path_m2=float(
                    active_area_per_path
                ),
                parallel_path_count=int(
                    parallel_path_count
                ),
                zone_count=int(zone_count_stage15),
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state[
                "coupled_drying_profile"
            ] = coupled_profile

            st.session_state[
                "coupled_drying_zone_profile"
            ] = coupled_zone_profile

            st.session_state[
                "coupled_drying_summary"
            ] = coupled_summary

            r1, r2, r3, r4 = st.columns(4)

            r1.metric(
                "Кінцева середня вологість",
                (
                    f"{coupled_summary['final_average_wb_pct']:.2f} "
                    "% w.b."
                ),
            )

            r2.metric(
                "Видалено води",
                (
                    f"{coupled_summary['total_water_removed_kg']:.3f} "
                    "кг"
                ),
            )

            r3.metric(
                "Середня фактична швидкість",
                (
                    f"{coupled_summary['mean_actual_removal_rate_kg_h']:.4f} "
                    "кг/год"
                ),
            )

            r4.metric(
                "Середня активна V̇вент",
                (
                    f"{coupled_summary['mean_fan_flow_total_m3_h']:.0f} "
                    "м³/год"
                ),
            )

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Паралельних трактів",
                str(coupled_summary["parallel_path_count"]),
            )

            p2.metric(
                "Витрата одного тракту",
                (
                    f"{coupled_summary['mean_fan_flow_per_path_m3_h']:.0f} "
                    "м³/год"
                ),
            )

            p3.metric(
                "Сумарна активна площа",
                (
                    f"{coupled_summary['active_flow_area_total_m2']:.3f} "
                    "м²"
                ),
            )

            if coupled_summary["target_reached"]:
                st.success(
                    "Усі розрахункові зони досягли заданої "
                    "кінцевої вологості. Після цього активний "
                    "потік сушіння автоматично вимикається."
                )

                if coupled_summary["target_time"]:
                    st.write(
                        "Момент досягнення цілі: "
                        f"**{coupled_summary['target_time']}**."
                    )
            else:
                st.warning(
                    "За заданої тривалості не всі зони досягли "
                    "кінцевої вологості."
                )

            st.markdown(
                "#### 6. Контроль екстраполяції за X₀"
            )

            e1, e2 = st.columns(2)

            e1.metric(
                "Вода в зоні екстраполяції X₀",
                (
                    f"{coupled_summary['source_extrapolated_removed_kg']:.3f} "
                    "кг"
                ),
            )

            e2.metric(
                "Частка від усієї видаленої води",
                (
                    f"{coupled_summary['source_extrapolated_removed_fraction_pct']:.1f} "
                    "%"
                ),
            )

            st.markdown(
                "#### 7. Обмеження здатністю повітря переносити вологу"
            )

            a1, a2 = st.columns(2)

            a1.metric(
                "Інтервали з обмеженням повітрям",
                str(
                    coupled_summary[
                        "air_capacity_limited_intervals"
                    ]
                ),
            )

            a2.metric(
                "Зменшення видалення води",
                (
                    f"{coupled_summary['air_capacity_limited_water_difference_kg']:.3f} "
                    "кг"
                ),
            )

            if (
                coupled_summary[
                    "page_rh_extrapolated_intervals"
                ] > 0
            ):
                st.warning(
                    "Є інтервали з екстраполяцією Page за RH "
                    "поза 20–60 %. Вони маркуються окремо."
                )

            if coupled_summary["unusable_intervals"] > 0:
                st.error(
                    "Є інтервали, де хоча б одна зона виходить "
                    "за допустимі T/v Page або область Modified Oswin."
                )

            st.markdown(
                "#### 8. Порівняння фактичної та цільової вологості"
            )

            comparison_df = pd.DataFrame(
                index=coupled_profile.index
            )

            comparison_df[
                "Зональна Page, % w.b."
            ] = coupled_profile[
                "zonal_average_moisture_end_wb_pct"
            ]

            if (
                "required_moisture_removal_profile"
                in st.session_state
            ):
                target_profile15 = st.session_state[
                    "required_moisture_removal_profile"
                ]

                if len(target_profile15) == len(comparison_df):
                    comparison_df[
                        "Цільова траєкторія, % w.b."
                    ] = target_profile15[
                        "target_product_moisture_wb_pct"
                    ].to_numpy()

            st.line_chart(
                comparison_df,
                use_container_width=True,
            )

            st.markdown(
                "#### 9. Фактична швидкість видалення води"
            )

            st.line_chart(
                coupled_profile[
                    ["actual_water_removal_rate_kg_h"]
                ],
                use_container_width=True,
            )

            st.markdown(
                "#### 10. Загальна продуктивність вентилятора"
            )

            st.line_chart(
                coupled_profile[
                    ["fan_volume_flow_total_m3_h"]
                ],
                use_container_width=True,
            )

            st.markdown(
                "#### 11. Зміна стану повітря вздовж тракту"
            )

            air_chart = pd.DataFrame(
                index=coupled_profile.index
            )

            air_chart["d вхід, г/кг"] = (
                coupled_profile[
                    "zonal_air_inlet_humidity_ratio_kg_kg"
                ] * 1000.0
            )

            air_chart["d вихід, г/кг"] = (
                coupled_profile[
                    "zonal_air_outlet_humidity_ratio_g_kg"
                ]
            )

            st.line_chart(
                air_chart,
                use_container_width=True,
            )

            st.markdown(
                "#### 12. Деталізація по зонах"
            )

            if not coupled_zone_profile.empty:
                st.dataframe(
                    coupled_zone_profile,
                    use_container_width=True,
                )

            st.download_button(
                "Завантажити часовий ряд зональної моделі",
                data=dataframe_to_csv_bytes(
                    coupled_profile
                ),
                file_name=(
                    "zonal_coupled_drying_airflow_linked.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_zonal_coupled_stage15_v12",
            )

            if not coupled_zone_profile.empty:
                st.download_button(
                    "Завантажити деталізацію по зонах",
                    data=dataframe_to_csv_bytes(
                        coupled_zone_profile
                    ),
                    file_name=(
                        "zonal_coupled_drying_airflow_linked_details.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                    key="download_zonal_details_stage15_v12",
                )

            st.warning(
                "Схема повітророзподілу є конструктивним вибором. "
                "Якщо реальна сушарка має горизонтальний обдув над "
                "лотками, а не проходження повітря крізь шар зерна, "
                "пряме перенесення v_Page із Ramaj потребуватиме "
                "окремого обґрунтування або експериментальної "
                "валідації."
            )

# ---------------------------------------------------------------------
# 11. ПЕРЕВІРКА ПРОДУКТИВНОСТІ СУШІННЯ
# ---------------------------------------------------------------------
with tab_performance:
    st.subheader(
        "Етап 16. Перевірка забезпечення потрібної "
        "швидкості видалення води"
    )

    st.info(
        "На цьому етапі цільова траєкторія з пункту 10 "
        "порівнюється з фактичною зональною кінетикою з пункту 15. "
        "Основним критерієм є не миттєва швидкість після дострокового "
        "завершення сушіння, а накопичувальне виконання заданого "
        "графіка видалення води."
    )

    st.markdown("#### Основні критерії")

    st.latex(
        r"\Delta\dot m_{\mathrm{в}}(t)"
        r"=\dot m_{\mathrm{в,факт}}(t)"
        r"-\dot m_{\mathrm{в,потр}}(t)"
    )

    st.latex(
        r"\Delta m_{\Sigma}(t)"
        r"=m_{\mathrm{в,факт},\Sigma}(t)"
        r"-m_{\mathrm{в,ціль},\Sigma}(t)"
    )

    st.latex(
        r"K_{\mathrm{прод}}"
        r"=\frac{\overline{\dot m}_{\mathrm{в,факт}}}"
        r"{\overline{\dot m}_{\mathrm{в,потр}}}"
    )

    st.latex(
        r"\Delta\tau"
        r"=\tau_{\mathrm{актив,потр}}"
        r"-\tau_{\mathrm{актив,факт}}"
    )

    st.write(
        "Якщо ΔmΣ(t)<0 до моменту досягнення кінцевої вологості, "
        "сушарка відстає від цільового графіка. Якщо кінцева "
        "вологість досягнута раніше, нульова швидкість після цього "
        "не вважається дефіцитом продуктивності."
    )

    if "coupled_drying_profile" not in st.session_state:
        st.warning(
            "Спочатку виконайте етап 15 у вкладці "
            "«Спільний розрахунок сушіння»."
        )

    elif (
        "required_moisture_removal_profile"
        not in st.session_state
    ):
        st.warning(
            "Спочатку сформуйте цільовий профіль у пункті 10."
        )

    else:
        actual16 = st.session_state[
            "coupled_drying_profile"
        ]

        target16 = st.session_state[
            "required_moisture_removal_profile"
        ]

        balance16 = calculate_mass_balance(
            float(initial_mass_kg),
            float(product["initial_moisture_pct"]),
            float(product["final_moisture_pct"]),
        )

        try:
            performance_profile, performance_summary = (
                evaluate_drying_performance_stage16(
                    coupled_profile=actual16,
                    target_profile=target16,
                    water_to_remove_kg=float(
                        balance16["water_to_remove_kg"]
                    ),
                    step_minutes=int(model_step_minutes),
                )
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state[
                "drying_performance_profile"
            ] = performance_profile

            st.session_state[
                "drying_performance_summary"
            ] = performance_summary

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Потрібна середня швидкість",
                (
                    f"{performance_summary['required_average_rate_kg_h']:.4f} "
                    "кг/год"
                ),
            )

            p2.metric(
                "Фактична середня швидкість",
                (
                    f"{performance_summary['actual_average_rate_kg_h']:.4f} "
                    "кг/год"
                ),
            )

            p3.metric(
                "Коефіцієнт продуктивності",
                (
                    f"{performance_summary['productivity_factor']:.2f}"
                ),
            )

            p4.metric(
                "Розрахунковий запас продуктивності",
                (
                    f"{performance_summary['productivity_reserve_pct']:.1f} %"
                ),
            )

            t1, t2, t3, t4 = st.columns(4)

            t1.metric(
                "Потрібний активний час",
                (
                    f"{performance_summary['required_active_hours']:.2f} "
                    "год"
                ),
            )

            t2.metric(
                "Фактичний активний час",
                (
                    f"{performance_summary['actual_active_hours']:.2f} "
                    "год"
                ),
            )

            t3.metric(
                "Запас активного часу",
                (
                    f"{performance_summary['active_time_reserve_h']:.2f} "
                    "год"
                ),
            )

            t4.metric(
                "Інтервали відставання від графіка",
                str(
                    performance_summary[
                        "behind_schedule_intervals"
                    ]
                ),
            )

            if performance_summary[
                "completion_criterion_met"
            ]:
                if performance_summary[
                    "schedule_never_behind_until_completion"
                ]:
                    st.success(
                        "Поточна модель прогнозує, що сушарка "
                        "забезпечує потрібну продуктивність і не "
                        "відстає від накопичувального цільового "
                        "графіка до моменту досягнення кінцевої "
                        "вологості."
                    )
                else:
                    st.warning(
                        "Кінцеву вологість досягнуто, але на окремих "
                        "інтервалах накопичувальний фактичний профіль "
                        "відставав від цільового графіка."
                    )
            else:
                st.error(
                    "За заданої тривалості процесу модель не забезпечує "
                    "видалення всієї потрібної маси води."
                )

            if performance_summary["completion_time"]:
                st.write(
                    "Розрахунковий момент завершення сушіння "
                    "(кінець останнього розрахункового інтервалу): "
                    f"**{performance_summary['completion_time']}**."
                )

            st.markdown(
                "#### Порівняння миттєвої швидкості"
            )

            rate_chart = performance_profile[
                [
                    "required_rate_kg_h",
                    "actual_rate_kg_h",
                ]
            ].rename(
                columns={
                    "required_rate_kg_h": (
                        "Потрібна швидкість, кг/год"
                    ),
                    "actual_rate_kg_h": (
                        "Фактична швидкість, кг/год"
                    ),
                }
            )

            st.line_chart(
                rate_chart,
                use_container_width=True,
            )

            st.caption(
                "Після дострокового досягнення 13 % фактична "
                "швидкість стає нульовою. Це не означає дефіцит: "
                "технологічна ціль уже виконана."
            )

            st.markdown(
                "#### Накопичувальне видалення води"
            )

            cumulative_chart = performance_profile[
                [
                    "target_cumulative_removed_kg",
                    "actual_cumulative_removed_kg",
                ]
            ].rename(
                columns={
                    "target_cumulative_removed_kg": (
                        "Цільове накопичувальне видалення, кг"
                    ),
                    "actual_cumulative_removed_kg": (
                        "Фактичне накопичувальне видалення, кг"
                    ),
                }
            )

            st.line_chart(
                cumulative_chart,
                use_container_width=True,
            )

            st.markdown(
                "#### Запас / дефіцит відносно цільового графіка"
            )

            st.line_chart(
                performance_profile[
                    ["cumulative_margin_kg"]
                ].rename(
                    columns={
                        "cumulative_margin_kg": (
                            "ΔmΣ, кг"
                        )
                    }
                ),
                use_container_width=True,
            )

            st.write(
                "Мінімальний накопичувальний запас до завершення: "
                f"**{performance_summary['minimum_cumulative_margin_kg']:.4f} кг**."
            )

            st.write(
                "Максимальне випередження цільового графіка: "
                f"**{performance_summary['maximum_cumulative_lead_kg']:.4f} кг**."
            )

            # Methodological warning from stage 15.
            if "coupled_drying_summary" in st.session_state:
                stage15_summary = st.session_state[
                    "coupled_drying_summary"
                ]

                extrapolation_share = float(
                    stage15_summary.get(
                        "source_extrapolated_removed_fraction_pct",
                        0.0,
                    )
                )

                if extrapolation_share > 0:
                    st.warning(
                        "Запас продуктивності вище є прогнозом "
                        "поточної математичної моделі, а не вже "
                        "експериментально підтвердженим запасом. "
                        f"Приблизно {extrapolation_share:.1f} % "
                        "видаленої води в етапі 15 припадає на "
                        "область екстраполяції Page за початковим "
                        "вологовмістом продукту."
                    )

            st.dataframe(
                performance_profile[
                    [
                        "operating_mode",
                        "required_rate_kg_h",
                        "actual_rate_kg_h",
                        "rate_difference_kg_h",
                        "target_cumulative_removed_kg",
                        "actual_cumulative_removed_kg",
                        "cumulative_margin_kg",
                        "target_moisture_wb_pct",
                        "actual_moisture_wb_pct",
                        "schedule_status",
                    ]
                ],
                use_container_width=True,
            )

            st.download_button(
                "Завантажити перевірку продуктивності",
                data=dataframe_to_csv_bytes(
                    performance_profile
                ),
                file_name=(
                    "stage16_drying_performance_check.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_stage16_performance",
            )

# ---------------------------------------------------------------------
# 12. НАГРІВАННЯ СУШИЛЬНОГО АГЕНТА
# ---------------------------------------------------------------------
with tab_air_heating:
    st.subheader(
        "Етап 17. Розрахунок теплоти на нагрівання "
        "сушильного агента"
    )

    st.info(
        "На цьому етапі визначається лише теплова потужність, "
        "потрібна для нагрівання зовнішнього вологого повітря "
        "до заданої температури сушіння перед входом у камеру. "
        "Теплота випаровування, нагрівання продукту, конструкцій, "
        "теплові втрати та рекуперація будуть додані окремо."
    )

    st.markdown("#### Психрометрична постановка")

    st.write(
        "При чутливому нагріванні сушильного агента перед камерою "
        "вологовміст не змінюється:"
    )

    st.latex(
        r"d_{\mathrm{нагр,вих}}"
        r"=d_{\mathrm{зовн}}"
    )

    st.write(
        "Ентальпія вологого повітря на 1 кг сухого повітря:"
    )

    st.latex(
        r"h=1.006T+d(2501+1.86T)"
    )

    st.write(
        "Потрібна теплова потужність для нагрівання:"
    )

    st.latex(
        r"\dot Q_{\mathrm{пов}}"
        r"=\dot m_{da}"
        r"\left("
        r"h_{\mathrm{суш}}-h_{\mathrm{зовн}}"
        r"\right)"
    )

    st.write(
        "Оскільки d до і після нагрівання однаковий, це рівнозначно:"
    )

    st.latex(
        r"\dot Q_{\mathrm{пов}}"
        r"=\dot m_{da}"
        r"(1.006+1.86d)"
        r"(T_{\mathrm{суш}}-T_{\mathrm{зовн}})"
    )

    st.caption(
        "h — кДж/кг сухого повітря; ṁda — кг сухого повітря/с; "
        "тому Q̇пов отримується у кВт."
    )

    st.warning(
        "Tсуш у поточній моделі ще є цільовою заданою температурою "
        "(для базової пшениці 40 °C), а не температурою, вже "
        "забезпеченою сонячним колектором. Тут ми визначаємо "
        "теплову ПОТРЕБУ. На подальших етапах вона буде порівняна "
        "з реально доступною тепловою генерацією."
    )

    if "coupled_drying_profile" not in st.session_state:
        st.warning(
            "Спочатку виконайте етап 15 у вкладці "
            "«Спільний розрахунок сушіння»."
        )

    else:
        stage17_input = st.session_state[
            "coupled_drying_profile"
        ]

        try:
            (
                air_heating_profile,
                air_heating_summary,
            ) = calculate_air_heating_demand_stage17(
                coupled_profile=stage17_input,
                step_minutes=int(model_step_minutes),
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state[
                "air_heating_profile"
            ] = air_heating_profile

            st.session_state[
                "air_heating_summary"
            ] = air_heating_summary

            q1, q2, q3, q4 = st.columns(4)

            q1.metric(
                "Пікова теплова потужність",
                (
                    f"{air_heating_summary['peak_air_heating_power_kw']:.2f} "
                    "кВт"
                ),
            )

            q2.metric(
                "Середня теплова потужність",
                (
                    f"{air_heating_summary['mean_air_heating_power_kw']:.2f} "
                    "кВт"
                ),
            )

            q3.metric(
                "Теплота на нагрівання повітря",
                (
                    f"{air_heating_summary['total_air_heating_energy_kwh']:.2f} "
                    "кВт·год"
                ),
            )

            q4.metric(
                "Тривалість активного нагрівання",
                (
                    f"{air_heating_summary['active_air_heating_hours']:.2f} "
                    "год"
                ),
            )

            s1, s2, s3 = st.columns(3)

            s1.metric(
                "Середній ΔT",
                (
                    f"{air_heating_summary['mean_air_heating_delta_t_c']:.2f} "
                    "°C"
                ),
            )

            s2.metric(
                "Середній Δh",
                (
                    f"{air_heating_summary['mean_air_heating_delta_h_kj_kg_da']:.2f} "
                    "кДж/кг с.п."
                ),
            )

            specific_q = air_heating_summary[
                "specific_air_heating_kwh_per_kg_removed_water"
            ]

            s3.metric(
                "На 1 кг видаленої води",
                (
                    f"{specific_q:.2f} кВт·год/кг"
                    if pd.notna(specific_q)
                    else "—"
                ),
            )

            if air_heating_summary[
                "cooling_intervals"
            ] > 0:
                st.warning(
                    "Є активні інтервали, де температура "
                    "зовнішнього повітря перевищує задану "
                    "температуру сушіння. Для них потреба в "
                    "нагріванні прийнята рівною нулю; окремий "
                    "розрахунок охолодження в цій роботі не "
                    "виконується."
                )

            if (
                air_heating_summary[
                    "max_enthalpy_cp_check_error_kw"
                ]
                > 1e-8
            ):
                st.warning(
                    "Виявлено розбіжність між ентальпійною "
                    "формою та перевіркою cp·ΔT. Перевірте "
                    "психрометричні дані."
                )
            else:
                st.success(
                    "Ентальпійний розрахунок і еквівалентна "
                    "форма через cp·ΔT узгоджуються."
                )

            st.markdown(
                "#### Температура зовнішнього і сушильного повітря"
            )

            temperature_chart = (
                air_heating_profile[
                    [
                        "air_heating_ambient_temp_c",
                        "air_heating_target_temp_c",
                    ]
                ]
                .rename(
                    columns={
                        "air_heating_ambient_temp_c": (
                            "T зовнішнього повітря, °C"
                        ),
                        "air_heating_target_temp_c": (
                            "T сушіння (цільова), °C"
                        ),
                    }
                )
            )

            st.line_chart(
                temperature_chart,
                use_container_width=True,
            )

            st.markdown(
                "#### Потрібна теплова потужність нагрівання повітря"
            )

            st.line_chart(
                air_heating_profile[
                    ["air_heating_power_kw"]
                ].rename(
                    columns={
                        "air_heating_power_kw": (
                            "Q̇пов, кВт"
                        )
                    }
                ),
                use_container_width=True,
            )

            st.markdown(
                "#### Накопичена теплота на нагрівання повітря"
            )

            st.line_chart(
                air_heating_profile[
                    ["air_heating_energy_kwh_cumulative"]
                ].rename(
                    columns={
                        "air_heating_energy_kwh_cumulative": (
                            "Qпов,Σ, кВт·год"
                        )
                    }
                ),
                use_container_width=True,
            )

            st.markdown(
                "#### Часовий ряд розрахунку"
            )

            stage17_columns = [
                "operating_mode",
                "air_heating_ambient_temp_c",
                "air_heating_target_temp_c",
                "air_heating_delta_t_c",
                "air_heating_humidity_ratio_kg_kg",
                "air_heating_cp_kj_kg_da_k",
                "air_heating_h_ambient_kj_kg_da",
                "air_heating_h_target_kj_kg_da",
                "air_heating_delta_h_kj_kg_da",
                "air_heating_dry_air_mass_flow_kg_h",
                "air_heating_power_kw",
                "air_heating_energy_kwh_interval",
                "air_heating_energy_kwh_cumulative",
                "active_drying_command",
            ]

            existing_stage17_columns = [
                col
                for col in stage17_columns
                if col in air_heating_profile.columns
            ]

            st.dataframe(
                air_heating_profile[
                    existing_stage17_columns
                ],
                use_container_width=True,
            )

            st.download_button(
                "Завантажити розрахунок нагрівання сушильного агента",
                data=dataframe_to_csv_bytes(
                    air_heating_profile
                ),
                file_name=(
                    "stage17_air_heating_demand.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_stage17_air_heating",
            )

# ---------------------------------------------------------------------
# 13. ЕНЕРГІЯ ФАЗОВОГО ПЕРЕХОДУ ВОДИ
# ---------------------------------------------------------------------
with tab_evaporation:
    st.subheader(
        "Етап 18. Розрахунок теплоти на випаровування вологи"
    )

    st.info(
        "На цьому етапі визначається енергія, фізично пов'язана "
        "з переходом видаленої з продукту води з рідкої фази "
        "у парову. Це окрема енергетична характеристика процесу."
    )

    st.markdown("#### Теплота фазового переходу")

    st.write(
        "Для водяної пари використовується наближене "
        "психрометричне співвідношення:"
    )

    st.latex(
        r"h_g(T)\approx2501+1.86T"
    )

    st.write(
        "Для рідкої води у розглянутому температурному діапазоні:"
    )

    st.latex(
        r"h_f(T)\approx4.186T"
    )

    st.write(
        "Тоді питома теплота пароутворення:"
    )

    st.latex(
        r"h_{fg}(T)"
        r"=h_g-h_f"
        r"\approx2501-2.326T"
    )

    st.write(
        "Енергія фазового переходу в окремій розрахунковій зоні:"
    )

    st.latex(
        r"\Delta Q_{\mathrm{вип},j}"
        r"=\Delta m_{\mathrm{в},j}"
        r"h_{fg}(T_{\mathrm{ref},j})"
    )

    st.write(
        "Середня потужність фазового переходу за один часовий крок:"
    )

    st.latex(
        r"\dot Q_{\mathrm{вип},i}"
        r"=\frac{\sum_j\Delta Q_{\mathrm{вип},i,j}}{\Delta t}"
    )

    st.warning(
        "Температура самого зерна ще не розрахована. Тому на цьому "
        "етапі Tref приймається як середня температура сушильного "
        "агента між входом і виходом відповідної зони. Це тимчасове "
        "наближення. Після розрахунку температури продукту в п.19 "
        "теплоту пароутворення потрібно буде уточнити."
    )

    st.error(
        "Qвип НЕ додається автоматично до Qпов із п.17. "
        "У поточній зональній моделі випаровування вже супроводжується "
        "зміною ентальпії сушильного агента. Просте складання "
        "Qпов + Qвип до формування повного енергетичного балансу "
        "може призвести до подвійного врахування тієї самої енергії."
    )

    if "coupled_drying_profile" not in st.session_state:
        st.warning(
            "Спочатку виконайте етап 15."
        )

    elif "coupled_drying_zone_profile" not in st.session_state:
        st.warning(
            "Для етапу 18 потрібна деталізація по зонах з етапу 15."
        )

    else:
        main18 = st.session_state[
            "coupled_drying_profile"
        ]

        zones18 = st.session_state[
            "coupled_drying_zone_profile"
        ]

        try:
            (
                evaporation_profile,
                evaporation_zone_profile,
                evaporation_summary,
            ) = calculate_evaporation_energy_stage18(
                coupled_profile=main18,
                zone_profile=zones18,
                step_minutes=int(model_step_minutes),
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state[
                "evaporation_energy_profile"
            ] = evaporation_profile

            st.session_state[
                "evaporation_energy_zone_profile"
            ] = evaporation_zone_profile

            st.session_state[
                "evaporation_energy_summary"
            ] = evaporation_summary

            e1, e2, e3, e4 = st.columns(4)

            e1.metric(
                "Випарувано води",
                (
                    f"{evaporation_summary['total_water_evaporated_kg']:.3f} "
                    "кг"
                ),
            )

            e2.metric(
                "Енергія фазового переходу",
                (
                    f"{evaporation_summary['total_evaporation_energy_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            e3.metric(
                "Пікова потужність випаровування",
                (
                    f"{evaporation_summary['peak_evaporation_power_kw']:.2f} "
                    "кВт"
                ),
            )

            e4.metric(
                "Середня потужність випаровування",
                (
                    f"{evaporation_summary['mean_evaporation_power_kw']:.2f} "
                    "кВт"
                ),
            )

            e5, e6, e7 = st.columns(3)

            e5.metric(
                "Середня Tref",
                (
                    f"{evaporation_summary['weighted_mean_reference_temp_c']:.2f} "
                    "°C"
                ),
            )

            e6.metric(
                "Середня hfg",
                (
                    f"{evaporation_summary['weighted_mean_latent_heat_kj_kg']:.1f} "
                    "кДж/кг"
                ),
            )

            e7.metric(
                "На 1 кг випаруваної води",
                (
                    f"{evaporation_summary['specific_evaporation_energy_kwh_per_kg']:.3f} "
                    "кВт·год/кг"
                ),
            )

            if (
                evaporation_summary[
                    "max_water_balance_error_kg"
                ]
                <= 1e-9
            ):
                st.success(
                    "Маса води у розрахунку фазового переходу "
                    "повністю узгоджується з масою води, видаленою "
                    "у зональній моделі етапу 15."
                )
            else:
                st.warning(
                    "Виявлено розбіжність між масою води у п.15 "
                    "і п.18. Максимальна похибка за інтервал: "
                    f"{evaporation_summary['max_water_balance_error_kg']:.6f} кг."
                )

            st.markdown(
                "#### Потужність фазового переходу"
            )

            st.line_chart(
                evaporation_profile[
                    ["evaporation_power_kw"]
                ].rename(
                    columns={
                        "evaporation_power_kw": (
                            "Q̇вип, кВт"
                        )
                    }
                ),
                use_container_width=True,
            )

            st.markdown(
                "#### Накопичена енергія фазового переходу"
            )

            st.line_chart(
                evaporation_profile[
                    ["evaporation_energy_kwh_cumulative"]
                ].rename(
                    columns={
                        "evaporation_energy_kwh_cumulative": (
                            "Qвип,Σ, кВт·год"
                        )
                    }
                ),
                use_container_width=True,
            )

            st.markdown(
                "#### Референсна температура та питома теплота пароутворення"
            )

            ref_chart = evaporation_profile[
                [
                    "evaporation_reference_temp_c",
                    "latent_heat_vaporization_kj_kg",
                ]
            ]

            st.dataframe(
                evaporation_profile[
                    [
                        "operating_mode",
                        "actual_water_removed_interval_kg",
                        "evaporation_water_removed_kg",
                        "evaporation_reference_temp_c",
                        "latent_heat_vaporization_kj_kg",
                        "evaporation_power_kw",
                        "evaporation_energy_kwh",
                        "evaporation_energy_kwh_cumulative",
                        "evaporation_water_balance_error_kg",
                    ]
                ],
                use_container_width=True,
            )

            st.markdown(
                "#### Деталізація фазового переходу по зонах"
            )

            st.dataframe(
                evaporation_zone_profile[
                    [
                        "time_position",
                        "zone",
                        "actual_water_removed_total_kg",
                        "air_inlet_temp_c",
                        "air_outlet_temp_c",
                        "evaporation_reference_temp_c",
                        "liquid_water_enthalpy_kj_kg",
                        "water_vapour_enthalpy_kj_kg",
                        "latent_heat_vaporization_kj_kg",
                        "evaporation_energy_kwh",
                        "evaporation_power_kw",
                    ]
                ],
                use_container_width=True,
            )

            st.download_button(
                "Завантажити розрахунок теплоти випаровування",
                data=dataframe_to_csv_bytes(
                    evaporation_profile
                ),
                file_name=(
                    "stage18_evaporation_energy.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_stage18_evaporation",
            )

            st.download_button(
                "Завантажити деталізацію випаровування по зонах",
                data=dataframe_to_csv_bytes(
                    evaporation_zone_profile
                ),
                file_name=(
                    "stage18_evaporation_energy_zones.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_stage18_evaporation_zones",
            )

# ---------------------------------------------------------------------
# 14. НАГРІВАННЯ ПРОДУКТУ
# ---------------------------------------------------------------------
with tab_product_heating:
    st.subheader(
        "Етап 19. Розрахунок теплоти на нагрівання продукту"
    )

    st.info(
        "На цьому етапі не задається штучна динамічна температура "
        "зерна. Для розрахунку Tₚ(t) потрібні коефіцієнт "
        "теплообміну повітря–зерно, питома поверхня та інші "
        "параметри повної системи з п.14. Натомість тут "
        "визначається прозора розрахункова теплота, необхідна "
        "для нагрівання початкового вологого продукту від Tₚ,0 "
        "до вибраної референсної температури Tₚ,ref."
    )

    st.markdown("#### Теплоємність вологої пшениці")

    st.latex(
        r"c_{p,\mathrm{пш}}"
        r"=(1-W)c_{p,\mathrm{сух}}"
        r"+Wc_{p,\mathrm{в}}"
    )

    st.write(
        "де W — масова частка води на вологій основі."
    )

    st.latex(
        r"c_{p,\mathrm{сух}}=1.550\ "
        r"\mathrm{kJ/(kg\cdot K)}"
    )

    st.latex(
        r"c_{p,\mathrm{в}}=4.186\ "
        r"\mathrm{kJ/(kg\cdot K)}"
    )

    st.caption(
        "Для сухої речовини пшениці використано значення "
        "1,550 кДж/(кг·К) за Kharchenko, Chornyi & Sharan (2020), "
        "Ukrainian Journal of Food Science, 8(1), 58–67, "
        "DOI 10.24263/2310-1008-2020-8-1-7."
    )

    st.markdown("#### Розрахункова теплота")

    st.latex(
        r"Q_{p,\mathrm{sens}}"
        r"=\left("
        r"m_{\mathrm{сух}}c_{p,\mathrm{сух}}"
        r"+m_{\mathrm{в},0}c_{p,\mathrm{в}}"
        r"\right)"
        r"\left(T_{p,\mathrm{ref}}-T_{p,0}\right)"
    )

    st.write(
        "Це еквівалентно формі "
        "Q = m₀ cₚ,волог ΔT для початкового вологого продукту."
    )

    st.warning(
        "Цей розрахунок є енергетичною оцінкою, а не прогнозом "
        "фактичної температурної траєкторії зерна. Значення "
        "Tₚ,ref можна трактувати як розрахунковий температурний "
        "рівень для оцінки sensible heat. За замовчуванням "
        "використовується цільова температура сушильного агента "
        "40 °C, тобто верхня консервативна оцінка для поточного етапу."
    )

    if "coupled_drying_profile" not in st.session_state:
        st.warning(
            "Спочатку виконайте етап 15."
        )
    else:
        coupled19 = st.session_state[
            "coupled_drying_profile"
        ]

        # Initial product temperature: by default equal to the ambient
        # temperature at the start of the process. User can override it.
        if (
            "T2M" in coupled19.columns
            and pd.notna(
                pd.to_numeric(
                    coupled19["T2M"],
                    errors="coerce",
                ).iloc[0]
            )
        ):
            default_initial_product_temp = float(
                pd.to_numeric(
                    coupled19["T2M"],
                    errors="coerce",
                ).iloc[0]
            )
        else:
            default_initial_product_temp = 20.0

        max_product_temp = float(
            product.get(
                "max_product_temp_c",
                product["target_drying_temp_c"],
            )
        )

        default_reference_temp = min(
            float(product["target_drying_temp_c"]),
            max_product_temp,
        )

        c1, c2 = st.columns(2)

        with c1:
            initial_product_temp_c = st.number_input(
                "Початкова температура продукту Tₚ,0, °C",
                min_value=-20.0,
                max_value=float(max_product_temp),
                value=float(default_initial_product_temp),
                step=0.5,
                format="%.2f",
                key="stage19_initial_product_temp_c",
            )

        with c2:
            min_reference_temp = float(
                initial_product_temp_c
            )

            safe_default_reference = max(
                min_reference_temp,
                float(default_reference_temp),
            )

            reference_product_temp_c = st.number_input(
                "Референсна температура продукту Tₚ,ref, °C",
                min_value=min_reference_temp,
                max_value=float(max_product_temp),
                value=float(safe_default_reference),
                step=0.5,
                format="%.2f",
                key="stage19_reference_product_temp_c",
            )

        if (
            abs(
                initial_product_temp_c
                - default_initial_product_temp
            )
            < 1e-9
        ):
            st.caption(
                "Початкову температуру автоматично прийнято рівною "
                "температурі зовнішнього повітря на початку процесу. "
                "Якщо фактична температура зерна перед завантаженням "
                "відома, її слід ввести вручну."
            )

        balance19 = calculate_mass_balance(
            float(initial_mass_kg),
            float(product["initial_moisture_pct"]),
            float(product["final_moisture_pct"]),
        )

        if "drying_performance_summary" in st.session_state:
            active_hours19 = float(
                st.session_state[
                    "drying_performance_summary"
                ]["actual_active_hours"]
            )
        else:
            active_hours19 = float(
                (
                    coupled19[
                        "actual_water_removed_interval_kg"
                    ] > 0
                ).sum()
                * int(model_step_minutes)
                / 60.0
            )

        try:
            (
                product_heat_profile,
                product_heat_sensitivity,
                product_heat_summary,
            ) = calculate_product_sensible_heat_stage19(
                coupled_profile=coupled19,
                dry_matter_kg=float(
                    balance19["dry_matter_kg"]
                ),
                initial_total_mass_kg=float(
                    initial_mass_kg
                ),
                initial_x_db=float(
                    balance19["initial_dry_basis"]
                ),
                target_x_db=float(
                    balance19["final_dry_basis"]
                ),
                initial_product_temp_c=float(
                    initial_product_temp_c
                ),
                reference_product_temp_c=float(
                    reference_product_temp_c
                ),
                active_drying_hours=float(
                    active_hours19
                ),
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state[
                "product_heating_profile"
            ] = product_heat_profile

            st.session_state[
                "product_heating_sensitivity"
            ] = product_heat_sensitivity

            st.session_state[
                "product_heating_summary"
            ] = product_heat_summary

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "cₚ початкового зерна",
                (
                    f"{product_heat_summary['initial_wet_product_cp_kj_kg_k']:.3f} "
                    "кДж/(кг·К)"
                ),
            )

            p2.metric(
                "cₚ при кінцевій вологості",
                (
                    f"{product_heat_summary['final_wet_product_cp_kj_kg_k']:.3f} "
                    "кДж/(кг·К)"
                ),
            )

            p3.metric(
                "ΔT продукту",
                (
                    f"{product_heat_summary['delta_t_c']:.2f} °C"
                ),
            )

            p4.metric(
                "Теплота на нагрівання продукту",
                (
                    f"{product_heat_summary['total_initial_product_sensible_heat_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            q1, q2, q3, q4 = st.columns(4)

            q1.metric(
                "Нагрівання сухої речовини",
                (
                    f"{product_heat_summary['dry_matter_sensible_heat_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            q2.metric(
                "Нагрівання всієї початкової води",
                (
                    f"{product_heat_summary['initial_water_sensible_heat_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            q3.metric(
                "З них вода, що буде випарувана",
                (
                    f"{product_heat_summary['removed_water_sensible_heat_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            q4.metric(
                "Еквівалентна середня потужність",
                (
                    f"{product_heat_summary['equivalent_mean_power_kw']:.3f} "
                    "кВт"
                ),
            )

            st.caption(
                "Еквівалентна середня потужність = Qₚ,нагр / "
                "фактичний активний час сушіння. Це лише нормалізація "
                "енергії за часом, а не реальна Q̇p(t)."
            )

            st.markdown(
                "#### Зміна питомої теплоємності продукту під час сушіння"
            )

            st.line_chart(
                product_heat_profile[
                    ["product_specific_heat_kj_kg_k"]
                ].rename(
                    columns={
                        "product_specific_heat_kj_kg_k": (
                            "cₚ продукту, кДж/(кг·К)"
                        )
                    }
                ),
                use_container_width=True,
            )

            st.markdown(
                "#### Чутливість Qₚ,sens до вибраної Tₚ,ref"
            )

            sensitivity_chart = (
                product_heat_sensitivity
                .set_index("product_reference_temp_c")[
                    ["product_sensible_heat_kwh"]
                ]
                .rename(
                    columns={
                        "product_sensible_heat_kwh": (
                            "Qₚ,sens, кВт·год"
                        )
                    }
                )
            )

            st.line_chart(
                sensitivity_chart,
                use_container_width=True,
            )

            st.markdown(
                "#### Складові теплоти на нагрівання продукту"
            )

            decomposition_df = pd.DataFrame(
                {
                    "Складова": [
                        "Суха речовина",
                        "Вся початкова вода",
                        "Вода, що залишається у кінцевому продукті",
                        "Вода, що буде випарувана",
                    ],
                    "Енергія, кВт·год": [
                        product_heat_summary[
                            "dry_matter_sensible_heat_kwh"
                        ],
                        product_heat_summary[
                            "initial_water_sensible_heat_kwh"
                        ],
                        (
                            product_heat_summary[
                                "final_product_sensible_heat_kwh"
                            ]
                            - product_heat_summary[
                                "dry_matter_sensible_heat_kwh"
                            ]
                        ),
                        product_heat_summary[
                            "removed_water_sensible_heat_kwh"
                        ],
                    ],
                }
            )

            st.dataframe(
                decomposition_df,
                use_container_width=True,
                hide_index=True,
            )

            st.warning(
                "Не слід автоматично складати Qₚ,sens, Qвип і Qпов "
                "до формування єдиного контрольного енергетичного "
                "балансу. На цьому етапі ці величини зберігаються "
                "окремо, щоб уникнути подвійного врахування енергії."
            )

            st.download_button(
                "Завантажити теплофізичний профіль продукту",
                data=dataframe_to_csv_bytes(
                    product_heat_profile
                ),
                file_name=(
                    "stage19_product_thermophysical_profile.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_stage19_product_profile",
            )

            st.download_button(
                "Завантажити чутливість теплоти до температури",
                data=dataframe_to_csv_bytes(
                    product_heat_sensitivity
                ),
                file_name=(
                    "stage19_product_heating_sensitivity.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_stage19_sensitivity",
            )

# ---------------------------------------------------------------------
# 15. ТЕПЛОВА ЄМНІСТЬ І СКЛАД КОНСТРУКЦІЙ СУШИЛЬНОЇ КАМЕРИ
# ---------------------------------------------------------------------
with tab_construction_heating:
    st.subheader(
        "Етап 20. Конструкція та теплова ємність сушильної камери"
    )

    st.info(
        "П.20 тепер є єдиним місцем, де описується конструкція "
        "сушильної камери. Ті самі шари огородження автоматично "
        "передаються в п.21 для розрахунку термічного опору та U. "
        "Лотки, рама та інші внутрішні елементи впливають на "
        "теплову ємність, але не входять у опір стінки."
    )

    st.latex(
        r"m_i=\rho_i A_i\delta_i"
    )

    st.latex(
        r"C_{\mathrm{констр}}=\sum_i m_i c_{p,i}"
    )

    st.write(
        "Для шарів огородження додатково зберігається "
        "теплопровідність λ, яка використовується вже в п.21:"
    )

    st.latex(
        r"R_{\mathrm{шар},i}=\frac{\delta_i}{\lambda_i}"
    )

    st.warning(
        "Кількість шарів повинна відповідати фактичній конструкції. "
        "Другий металевий лист, двері або інша окрема ділянка "
        "не додаються програмою автоматично. Якщо вони реально є, "
        "їх потрібно описати тут."
    )

    st.markdown("#### Введення конструкції")

    st.caption(
        "Усі поля знаходяться у формі: зміни застосовуються лише "
        "після натискання «Зберегти дані п.20». "
        "Десятковий роздільник — крапка."
    )

    construction_mass_mode = st.radio(
        "Спосіб визначення маси",
        options=[
            "Розрахувати масу за геометрією і густиною",
            "Ввести фактичну масу елементів",
        ],
        index=0,
        key="stage20_construction_mass_mode_v4",
    )

    components20 = None
    stage20_saved_now = False

    if "dryer_geometry" in st.session_state:
        geometry20 = st.session_state["dryer_geometry"]

        chamber_l20 = float(
            geometry20.get("chamber_length_m", 0.0)
        )
        chamber_w20 = float(
            geometry20.get("chamber_width_m", 0.0)
        )
        chamber_h20 = float(
            geometry20.get("chamber_height_m", 0.0)
        )

        enclosure_area20 = (
            2.0
            * (
                chamber_l20 * chamber_w20
                + chamber_l20 * chamber_h20
                + chamber_w20 * chamber_h20
            )
        )

        trays_area20 = float(
            geometry20.get(
                "total_drying_area_m2",
                0.0,
            )
        )
    else:
        enclosure_area20 = 0.0
        trays_area20 = 0.0

    if (
        construction_mass_mode
        == "Розрахувати масу за геометрією і густиною"
    ):
        if "dryer_geometry" not in st.session_state:
            st.warning(
                "Спочатку задайте геометрію сушильної камери "
                "у вкладці 6."
            )
        else:
            g1, g2 = st.columns(2)

            g1.metric(
                "Геометрична площа 6 поверхонь камери",
                f"{enclosure_area20:.3f} м²",
            )

            g2.metric(
                "Сумарна площа сушильних лотків",
                f"{trays_area20:.3f} м²",
            )

            default_geometry20 = pd.DataFrame(
                [
                    {
                        "include": False,
                        "element_role": "Шар огородження",
                        "assembly": "Основне огородження",
                        "component": "Внутрішній матеріальний шар",
                        "area_m2": enclosure_area20,
                        "thickness_mm": 0.0,
                        "density_kg_m3": 0.0,
                        "cp_kj_kg_k": 0.0,
                        "thermal_conductivity_w_m_k": 0.0,
                        "source_note": "",
                    },
                    {
                        "include": False,
                        "element_role": "Шар огородження",
                        "assembly": "Основне огородження",
                        "component": "Теплоізоляція",
                        "area_m2": enclosure_area20,
                        "thickness_mm": 0.0,
                        "density_kg_m3": 0.0,
                        "cp_kj_kg_k": 0.0,
                        "thermal_conductivity_w_m_k": 0.0,
                        "source_note": "",
                    },
                    {
                        "include": False,
                        "element_role": "Шар огородження",
                        "assembly": "Основне огородження",
                        "component": "Зовнішній матеріальний шар",
                        "area_m2": enclosure_area20,
                        "thickness_mm": 0.0,
                        "density_kg_m3": 0.0,
                        "cp_kj_kg_k": 0.0,
                        "thermal_conductivity_w_m_k": 0.0,
                        "source_note": "",
                    },
                    {
                        "include": False,
                        "element_role": "Внутрішній елемент",
                        "assembly": "",
                        "component": "Лотки",
                        "area_m2": trays_area20,
                        "thickness_mm": 0.0,
                        "density_kg_m3": 0.0,
                        "cp_kj_kg_k": 0.0,
                        "thermal_conductivity_w_m_k": 0.0,
                        "source_note": "",
                    },
                    {
                        "include": False,
                        "element_role": "Внутрішній елемент",
                        "assembly": "",
                        "component": "Рама / інше",
                        "area_m2": 0.0,
                        "thickness_mm": 0.0,
                        "density_kg_m3": 0.0,
                        "cp_kj_kg_k": 0.0,
                        "thermal_conductivity_w_m_k": 0.0,
                        "source_note": "",
                    },
                ]
            )

            old_geometry20 = st.session_state.get(
                "stage20_saved_geometry_components_raw",
                default_geometry20,
            )

            geometry_form_source20 = (
                upgrade_stage20_components_for_linkage(
                    old_geometry20
                )
            )

            with st.form(
                "stage20_geometry_input_form_v4",
                clear_on_submit=False,
            ):
                geometry_editor20 = st.data_editor(
                    geometry_form_source20,
                    num_rows="dynamic",
                    width="stretch",
                    hide_index=True,
                    key="stage20_geometry_components_editor_v4",
                    column_config={
                        "include": st.column_config.CheckboxColumn(
                            "Враховувати",
                        ),
                        "element_role": st.column_config.SelectboxColumn(
                            "Роль елемента",
                            options=[
                                "Шар огородження",
                                "Внутрішній елемент",
                            ],
                            required=True,
                        ),
                        "assembly": st.column_config.TextColumn(
                            "Вузол огородження",
                            help=(
                                "Для послідовних шарів однієї стінки "
                                "вкажіть однакову назву. Окремі двері "
                                "задавайте окремим вузлом."
                            ),
                        ),
                        "component": st.column_config.TextColumn(
                            "Елемент / матеріал",
                        ),
                        "area_m2": st.column_config.NumberColumn(
                            "A, м²",
                            min_value=0.0,
                            step=0.01,
                            format="%.3f",
                        ),
                        "thickness_mm": st.column_config.NumberColumn(
                            "δ, мм",
                            min_value=0.0,
                            step=0.01,
                            format="%.2f",
                        ),
                        "density_kg_m3": st.column_config.NumberColumn(
                            "ρ, кг/м³",
                            min_value=0.0,
                            step=0.01,
                            format="%.2f",
                        ),
                        "cp_kj_kg_k": st.column_config.NumberColumn(
                            "cₚ, кДж/(кг·К)",
                            min_value=0.0,
                            step=0.001,
                            format="%.3f",
                        ),
                        "thermal_conductivity_w_m_k": (
                            st.column_config.NumberColumn(
                                "λ, Вт/(м·К)",
                                min_value=0.0,
                                step=0.001,
                                format="%.4f",
                                help=(
                                    "Потрібна для шарів огородження "
                                    "у п.21. Для лотків/рами може "
                                    "залишатися 0."
                                ),
                            )
                        ),
                        "source_note": st.column_config.TextColumn(
                            "Джерело / примітка",
                        ),
                    },
                )

                geometry_submit20 = st.form_submit_button(
                    "Зберегти дані п.20",
                    type="primary",
                    use_container_width=True,
                )

            if geometry_submit20:
                geometry_saved20 = (
                    upgrade_stage20_components_for_linkage(
                        geometry_editor20
                    )
                )

                geometry_saved20["mass_kg"] = (
                    pd.to_numeric(
                        geometry_saved20["area_m2"],
                        errors="coerce",
                    ).fillna(0.0)
                    * (
                        pd.to_numeric(
                            geometry_saved20["thickness_mm"],
                            errors="coerce",
                        ).fillna(0.0)
                        / 1000.0
                    )
                    * pd.to_numeric(
                        geometry_saved20["density_kg_m3"],
                        errors="coerce",
                    ).fillna(0.0)
                )

                st.session_state[
                    "stage20_saved_geometry_components_raw"
                ] = geometry_editor20.copy()

                st.session_state[
                    "stage20_saved_geometry_components"
                ] = geometry_saved20.copy()

                st.session_state[
                    "stage20_saved_components_active"
                ] = geometry_saved20.copy()

                st.session_state[
                    "stage20_saved_mass_mode"
                ] = construction_mass_mode

                stage20_saved_now = True

            if (
                st.session_state.get(
                    "stage20_saved_mass_mode"
                )
                == construction_mass_mode
                and "stage20_saved_geometry_components"
                in st.session_state
            ):
                components20 = (
                    upgrade_stage20_components_for_linkage(
                        st.session_state[
                            "stage20_saved_geometry_components"
                        ]
                    )
                )

    else:
        default_direct20 = pd.DataFrame(
            [
                {
                    "include": False,
                    "element_role": "Внутрішній елемент",
                    "assembly": "",
                    "component": "Елемент",
                    "mass_kg": 0.0,
                    "cp_kj_kg_k": 0.0,
                    "area_m2": 0.0,
                    "thickness_mm": 0.0,
                    "thermal_conductivity_w_m_k": 0.0,
                    "source_note": "",
                },
            ]
        )

        old_direct20 = st.session_state.get(
            "stage20_saved_direct_components",
            default_direct20,
        )

        direct_form_source20 = (
            upgrade_stage20_components_for_linkage(
                old_direct20
            )
        )

        st.caption(
            "Якщо елемент є шаром огородження, для зв'язку з п.21 "
            "потрібно також вказати A, δ і λ. Для внутрішніх елементів "
            "ці поля можуть залишатися нульовими."
        )

        with st.form(
            "stage20_direct_input_form_v4",
            clear_on_submit=False,
        ):
            direct_editor20 = st.data_editor(
                direct_form_source20,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="stage20_direct_components_editor_v4",
                column_config={
                    "include": st.column_config.CheckboxColumn(
                        "Враховувати",
                    ),
                    "element_role": st.column_config.SelectboxColumn(
                        "Роль елемента",
                        options=[
                            "Шар огородження",
                            "Внутрішній елемент",
                        ],
                        required=True,
                    ),
                    "assembly": st.column_config.TextColumn(
                        "Вузол огородження",
                    ),
                    "component": st.column_config.TextColumn(
                        "Елемент / матеріал",
                    ),
                    "mass_kg": st.column_config.NumberColumn(
                        "m, кг",
                        min_value=0.0,
                        step=0.01,
                        format="%.3f",
                    ),
                    "cp_kj_kg_k": st.column_config.NumberColumn(
                        "cₚ, кДж/(кг·К)",
                        min_value=0.0,
                        step=0.001,
                        format="%.3f",
                    ),
                    "area_m2": st.column_config.NumberColumn(
                        "A, м²",
                        min_value=0.0,
                        step=0.01,
                        format="%.3f",
                    ),
                    "thickness_mm": st.column_config.NumberColumn(
                        "δ, мм",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                    ),
                    "thermal_conductivity_w_m_k": (
                        st.column_config.NumberColumn(
                            "λ, Вт/(м·К)",
                            min_value=0.0,
                            step=0.001,
                            format="%.4f",
                        )
                    ),
                    "source_note": st.column_config.TextColumn(
                        "Джерело / примітка",
                    ),
                },
            )

            direct_submit20 = st.form_submit_button(
                "Зберегти дані п.20",
                type="primary",
                use_container_width=True,
            )

        if direct_submit20:
            direct_saved20 = (
                upgrade_stage20_components_for_linkage(
                    direct_editor20
                )
            )

            st.session_state[
                "stage20_saved_direct_components"
            ] = direct_saved20.copy()

            st.session_state[
                "stage20_saved_components_active"
            ] = direct_saved20.copy()

            st.session_state[
                "stage20_saved_mass_mode"
            ] = construction_mass_mode

            stage20_saved_now = True

        if (
            st.session_state.get(
                "stage20_saved_mass_mode"
            )
            == construction_mass_mode
            and "stage20_saved_direct_components"
            in st.session_state
        ):
            components20 = (
                upgrade_stage20_components_for_linkage(
                    st.session_state[
                        "stage20_saved_direct_components"
                    ]
                )
            )

    if stage20_saved_now:
        st.success(
            "Дані п.20 збережено. П.21 автоматично використовуватиме "
            "ті самі активні шари огородження."
        )

    if components20 is None:
        st.info(
            "Заповніть потрібні поля та натисніть "
            "«Зберегти дані п.20»."
        )

    else:
        try:
            (
                construction_result20,
                construction_summary20,
            ) = calculate_construction_thermal_capacity_stage20(
                components_df=components20,
            )

        except Exception as exc:
            st.warning(str(exc))

        else:
            st.session_state[
                "construction_thermal_capacity_components"
            ] = construction_result20

            st.session_state[
                "construction_thermal_capacity_summary"
            ] = construction_summary20

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Маса врахованих конструкцій",
                (
                    f"{construction_summary20['total_construction_mass_kg']:.2f} "
                    "кг"
                ),
            )

            c2.metric(
                "Σmcₚ",
                (
                    f"{construction_summary20['total_thermal_capacity_kj_k']:.2f} "
                    "кДж/К"
                ),
            )

            c3.metric(
                "Теплова ємність",
                (
                    f"{construction_summary20['energy_per_1k_kwh']:.4f} "
                    "кВт·год/К"
                ),
            )

            c4.metric(
                "Середньозважене cₚ",
                (
                    f"{construction_summary20['weighted_cp_kj_kg_k']:.3f} "
                    "кДж/(кг·К)"
                ),
            )

            st.markdown(
                "#### Збережені елементи конструкції"
            )

            display_cols20 = [
                "element_role",
                "assembly",
                "component",
                "area_m2",
                "thickness_mm",
                "mass_kg",
                "density_kg_m3",
                "cp_kj_kg_k",
                "thermal_conductivity_w_m_k",
                "thermal_capacity_kj_k",
                "source_note",
            ]

            existing_cols20 = [
                col
                for col in display_cols20
                if col in construction_result20.columns
            ]

            stage20_display = construction_result20.loc[
                construction_result20["include"],
                existing_cols20,
            ].copy()

            stage20_display = stage20_display.rename(
                columns={
                    "element_role": "Роль елемента",
                    "assembly": "Вузол огородження",
                    "component": "Елемент / матеріал",
                    "area_m2": "A, м²",
                    "thickness_mm": "δ, мм",
                    "mass_kg": "m, кг",
                    "density_kg_m3": "ρ, кг/м³",
                    "cp_kj_kg_k": "cₚ, кДж/(кг·К)",
                    "thermal_conductivity_w_m_k": "λ, Вт/(м·К)",
                    "thermal_capacity_kj_k": "Cᵢ, кДж/К",
                    "source_note": "Джерело / примітка",
                }
            )

            st.dataframe(
                stage20_display,
                width="stretch",
                hide_index=True,
            )

            try:
                (
                    envelope_layers20,
                    envelope_assemblies20,
                ) = build_envelope_assemblies_stage21(
                    construction_result20
                )

            except Exception as exc:
                st.info(
                    "Теплова ємність п.20 розрахована. "
                    "Для автоматичного переходу до п.21: "
                    + str(exc)
                )

            else:
                st.session_state[
                    "stage20_envelope_layers"
                ] = envelope_layers20

                st.session_state[
                    "stage20_envelope_assemblies"
                ] = envelope_assemblies20

                st.markdown(
                    "#### Що автоматично передається в п.21"
                )

                stage20_to21_display = envelope_assemblies20[
                    [
                        "assembly",
                        "area_m2",
                        "layer_count",
                        "r_layers_m2_k_w",
                        "u_layers_w_m2_k",
                    ]
                ].copy().rename(
                    columns={
                        "assembly": "Вузол огородження",
                        "area_m2": "A, м²",
                        "layer_count": "Кількість шарів",
                        "r_layers_m2_k_w": "Rшарів, м²·К/Вт",
                        "u_layers_w_m2_k": "1/Rшарів, Вт/(м²·К)",
                    }
                )

                st.dataframe(
                    stage20_to21_display,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "1/Rшарів показано лише як характеристику "
                    "теплопровідності пакета шарів. Це ще не повний "
                    "U від внутрішнього повітря до зовнішнього, "
                    "бо поверхневі опори не включені."
                )

            st.download_button(
                "Завантажити дані конструкції",
                data=dataframe_to_csv_bytes(
                    construction_result20
                ),
                file_name=(
                    "stage20_construction_and_capacity.csv"
                ),
                mime="text/csv",
                width="stretch",
                key="download_stage20_capacity_v4",
            )

# ---------------------------------------------------------------------
# 16. ТЕПЛОВІ ВТРАТИ СУШИЛЬНОЇ КАМЕРИ
# ---------------------------------------------------------------------
with tab_heat_losses:
    st.subheader(
        "Етап 21. Теплопередача через огородження та теплові втрати"
    )

    st.info(
        "П.21 повністю використовує дані попередніх етапів. "
        "Шари та їх R беруться з п.20; внутрішня швидкість повітря "
        "визначається з витрати вентилятора і перерізу камери; "
        "зовнішня швидкість — з NASA POWER WS10M. "
        "hвн, hзовн та U вручну не задаються."
    )

    st.markdown("#### Розрахункова схема")

    st.latex(
        r"\mathrm{Re}_L="
        r"\frac{\rho V L}{\mu}"
    )

    st.latex(
        r"\overline{\mathrm{Nu}}_L="
        r"0.664\,\mathrm{Re}_L^{1/2}"
        r"\mathrm{Pr}^{1/3},"
        r"\qquad \mathrm{Re}_L\le5\cdot10^5"
    )

    st.latex(
        r"\overline{\mathrm{Nu}}_L="
        r"\left(0.037\,\mathrm{Re}_L^{0.8}-871\right)"
        r"\mathrm{Pr}^{1/3},"
        r"\qquad \mathrm{Re}_L>5\cdot10^5"
    )

    st.latex(
        r"h=\frac{\overline{\mathrm{Nu}}_L\,\lambda_{\mathrm{air}}}{L}"
    )

    st.latex(
        r"U_j(t)=\left["
        r"\frac{1}{h_{\mathrm{вн}}(t)}"
        r"+R_{\mathrm{шарів},j}"
        r"+\frac{1}{h_{\mathrm{зовн}}(t)}"
        r"\right]^{-1}"
    )

    st.latex(
        r"\dot Q_{\mathrm{втр}}(t)"
        r"=\sum_j U_j(t)A_j"
        r"\left[T_{\mathrm{кам}}(t)-T_{\mathrm{зовн}}(t)\right]"
    )

    st.caption(
        "Для поверхневого теплообміну використовується кореляція "
        "середнього Nusselt для примусового обтікання плоскої "
        "поверхні. Для внутрішньої сторони характерна довжина — "
        "довжина основного шляху сушильного агента; для зовнішньої — "
        "найбільший габарит камери."
    )

    st.warning(
        "NASA POWER надає WS10M на висоті 10 м. У цій версії "
        "швидкість не перераховується до фактичної висоти сушарки, "
        "оскільки для такого перерахунку потрібні додаткові параметри "
        "шорсткості місцевості. Тому hзовн є розрахунковим "
        "коефіцієнтом примусової конвекції на основі WS10M. "
        "Природна конвекція та радіаційний теплообмін тут окремо "
        "не додаються."
    )

    if (
        "construction_thermal_capacity_components"
        not in st.session_state
    ):
        st.warning(
            "Спочатку збережіть конструкцію у п.20."
        )

    elif "coupled_drying_profile" not in st.session_state:
        st.warning(
            "Спочатку виконайте спільний розрахунок сушіння."
        )

    elif "coupled_drying_zone_profile" not in st.session_state:
        st.warning(
            "Для п.21 потрібен зональний профіль температури "
            "та вологовмісту повітря."
        )

    elif "dryer_geometry" not in st.session_state:
        st.warning(
            "Для п.21 потрібна геометрія сушильної камери."
        )

    else:
        construction21 = (
            st.session_state[
                "construction_thermal_capacity_components"
            ]
        )

        try:
            (
                envelope_layers21,
                envelope_assemblies21,
            ) = build_envelope_assemblies_stage21(
                construction21
            )

        except Exception as exc:
            st.warning(str(exc))

        else:
            st.markdown(
                "#### Шари огородження з п.20"
            )

            stage21_layers_display = (
                envelope_layers21[
                    [
                        "assembly",
                        "component",
                        "area_m2",
                        "thickness_mm",
                        "thermal_conductivity_w_m_k",
                        "layer_resistance_m2_k_w",
                        "source_note",
                    ]
                ]
                .copy()
                .rename(
                    columns={
                        "assembly": "Вузол огородження",
                        "component": "Шар / матеріал",
                        "area_m2": "A, м²",
                        "thickness_mm": "δ, мм",
                        "thermal_conductivity_w_m_k": (
                            "λ, Вт/(м·К)"
                        ),
                        "layer_resistance_m2_k_w": (
                            "Rᵢ, м²·К/Вт"
                        ),
                        "source_note": (
                            "Джерело / примітка"
                        ),
                    }
                )
            )

            st.dataframe(
                stage21_layers_display,
                width="stretch",
                hide_index=True,
            )

            st.markdown(
                "#### Опір матеріальних шарів"
            )

            stage21_assemblies_display = (
                envelope_assemblies21[
                    [
                        "assembly",
                        "area_m2",
                        "layer_count",
                        "r_layers_m2_k_w",
                    ]
                ]
                .copy()
                .rename(
                    columns={
                        "assembly": "Вузол огородження",
                        "area_m2": "A, м²",
                        "layer_count": "Кількість шарів",
                        "r_layers_m2_k_w": (
                            "Rшарів, м²·К/Вт"
                        ),
                    }
                )
            )

            st.dataframe(
                stage21_assemblies_display,
                width="stretch",
                hide_index=True,
            )

            try:
                (
                    heat_loss_profile21,
                    heat_loss_envelope21,
                    heat_loss_summary21,
                ) = calculate_transmission_heat_losses_stage21(
                    coupled_profile=st.session_state[
                        "coupled_drying_profile"
                    ],
                    zone_profile=st.session_state[
                        "coupled_drying_zone_profile"
                    ],
                    envelope_assemblies_df=(
                        envelope_assemblies21
                    ),
                    geometry=st.session_state[
                        "dryer_geometry"
                    ],
                    step_minutes=int(
                        model_step_minutes
                    ),
                )

            except Exception as exc:
                st.warning(str(exc))

            else:
                st.session_state[
                    "heat_loss_profile"
                ] = heat_loss_profile21

                st.session_state[
                    "heat_loss_envelope"
                ] = heat_loss_envelope21

                st.session_state[
                    "heat_loss_summary"
                ] = heat_loss_summary21

                st.markdown(
                    "#### Автоматично визначений поверхневий теплообмін"
                )

                a1, a2, a3, a4 = st.columns(4)

                a1.metric(
                    "Середній hвн",
                    (
                        f"{heat_loss_summary21['mean_h_inside_w_m2_k']:.3f} "
                        "Вт/(м²·К)"
                    ),
                )

                a2.metric(
                    "Середній hзовн",
                    (
                        f"{heat_loss_summary21['mean_h_outside_w_m2_k']:.3f} "
                        "Вт/(м²·К)"
                    ),
                )

                a3.metric(
                    "Lвн",
                    (
                        f"{heat_loss_summary21['inside_characteristic_length_m']:.3f} "
                        "м"
                    ),
                )

                a4.metric(
                    "Lзовн",
                    (
                        f"{heat_loss_summary21['outside_characteristic_length_m']:.3f} "
                        "м"
                    ),
                )

                st.caption(
                    "Vвн визначається як повна витрата вентилятора, "
                    "поділена на геометричний переріз камери. "
                    "Vзовн = WS10M із погодного ряду NASA POWER."
                )

                r1, r2, r3, r4 = st.columns(4)

                r1.metric(
                    "Середній Reвн",
                    (
                        f"{heat_loss_summary21['mean_re_inside']:.0f}"
                    ),
                )

                r2.metric(
                    "Середній Nuвн",
                    (
                        f"{heat_loss_summary21['mean_nu_inside']:.2f}"
                    ),
                )

                r3.metric(
                    "Середній Reзовн",
                    (
                        f"{heat_loss_summary21['mean_re_outside']:.0f}"
                    ),
                )

                r4.metric(
                    "Середній Nuзовн",
                    (
                        f"{heat_loss_summary21['mean_nu_outside']:.2f}"
                    ),
                )

                st.markdown(
                    "#### Коефіцієнт теплопередачі огородження"
                )

                stage21_u_display = (
                    heat_loss_envelope21.copy().rename(
                        columns={
                            "assembly": "Вузол огородження",
                            "area_m2": "A, м²",
                            "r_layers_m2_k_w": (
                                "Rшарів, м²·К/Вт"
                            ),
                            "mean_h_inside_w_m2_k": (
                                "hвн,сер, Вт/(м²·К)"
                            ),
                            "mean_h_outside_w_m2_k": (
                                "hзовн,сер, Вт/(м²·К)"
                            ),
                            "mean_r_inside_m2_k_w": (
                                "Rвн,сер, м²·К/Вт"
                            ),
                            "mean_r_outside_m2_k_w": (
                                "Rзовн,сер, м²·К/Вт"
                            ),
                            "mean_u_w_m2_k": (
                                "Uсер, Вт/(м²·К)"
                            ),
                            "min_u_w_m2_k": (
                                "Umin, Вт/(м²·К)"
                            ),
                            "max_u_w_m2_k": (
                                "Umax, Вт/(м²·К)"
                            ),
                        }
                    )
                )

                st.dataframe(
                    stage21_u_display,
                    width="stretch",
                    hide_index=True,
                )

                b1, b2, b3, b4 = st.columns(4)

                b1.metric(
                    "Середньозважений U",
                    (
                        f"{heat_loss_summary21['mean_area_weighted_u_w_m2_k']:.3f} "
                        "Вт/(м²·К)"
                    ),
                )

                b2.metric(
                    "Пікові теплові втрати",
                    (
                        f"{heat_loss_summary21['peak_transmission_heat_loss_kw']:.3f} "
                        "кВт"
                    ),
                )

                b3.metric(
                    "Середні теплові втрати",
                    (
                        f"{heat_loss_summary21['mean_transmission_heat_loss_kw']:.3f} "
                        "кВт"
                    ),
                )

                if pd.notna(
                    heat_loss_summary21[
                        "total_transmission_heat_loss_kwh"
                    ]
                ):
                    total_loss_text21 = (
                        f"{heat_loss_summary21['total_transmission_heat_loss_kwh']:.3f} "
                        "кВт·год"
                    )
                else:
                    total_loss_text21 = "не визначено"

                b4.metric(
                    "Енергія теплових втрат",
                    total_loss_text21,
                )

                missing21 = int(
                    heat_loss_summary21[
                        "missing_active_interval_count"
                    ]
                )

                if missing21 == 0:
                    st.success(
                        "hвн(t), hзовн(t), U(t) та теплові втрати "
                        "визначені для всіх активних інтервалів сушіння."
                    )
                else:
                    st.warning(
                        f"Для {missing21} активних інтервалів "
                        "не вдалося визначити поверхневий теплообмін. "
                        "Сумарна енергія втрат для такого ряду "
                        "навмисно не підміняється неповним результатом."
                    )

                st.markdown(
                    "#### Швидкості та коефіцієнти теплообміну"
                )

                st.write(
                    "**Внутрішня швидкість повітря**"
                )

                st.line_chart(
                    heat_loss_profile21[
                        [
                            "heat_loss_inside_bulk_velocity_m_s",
                        ]
                    ].rename(
                        columns={
                            "heat_loss_inside_bulk_velocity_m_s": (
                                "Vвн, м/с"
                            ),
                        }
                    ),
                    width="stretch",
                )

                st.write(
                    "**Зовнішня швидкість вітру NASA POWER**"
                )

                st.line_chart(
                    heat_loss_profile21[
                        [
                            "heat_loss_ws10m_m_s",
                        ]
                    ].rename(
                        columns={
                            "heat_loss_ws10m_m_s": (
                                "WS10M, м/с"
                            ),
                        }
                    ),
                    width="stretch",
                )

                st.write(
                    "**Коефіцієнти поверхневого теплообміну**"
                )

                st.line_chart(
                    heat_loss_profile21[
                        [
                            "h_inside_w_m2_k",
                            "h_outside_w_m2_k",
                        ]
                    ].rename(
                        columns={
                            "h_inside_w_m2_k": (
                                "hвн, Вт/(м²·К)"
                            ),
                            "h_outside_w_m2_k": (
                                "hзовн, Вт/(м²·К)"
                            ),
                        }
                    ),
                    width="stretch",
                )

                st.markdown(
                    "#### Температурний напір"
                )

                st.line_chart(
                    heat_loss_profile21[
                        [
                            "heat_loss_chamber_temp_c",
                            "heat_loss_ambient_temp_c",
                        ]
                    ].rename(
                        columns={
                            "heat_loss_chamber_temp_c": (
                                "Tкам, °C"
                            ),
                            "heat_loss_ambient_temp_c": (
                                "Tзовн, °C"
                            ),
                        }
                    ),
                    width="stretch",
                )

                st.markdown(
                    "#### Теплова потужність втрат"
                )

                st.line_chart(
                    heat_loss_profile21[
                        ["heat_loss_power_kw"]
                    ].rename(
                        columns={
                            "heat_loss_power_kw": (
                                "Q̇втр, кВт"
                            )
                        }
                    ),
                    width="stretch",
                )

                st.markdown(
                    "#### Накопичена теплова енергія втрат"
                )

                st.line_chart(
                    heat_loss_profile21[
                        [
                            "heat_loss_energy_kwh_cumulative"
                        ]
                    ].rename(
                        columns={
                            "heat_loss_energy_kwh_cumulative": (
                                "Qвтр,Σ, кВт·год"
                            )
                        }
                    ),
                    width="stretch",
                )

                st.download_button(
                    "Завантажити часовий ряд п.21",
                    data=dataframe_to_csv_bytes(
                        heat_loss_profile21
                    ),
                    file_name=(
                        "stage21_heat_transfer_and_losses.csv"
                    ),
                    mime="text/csv",
                    width="stretch",
                    key="download_stage21_heat_losses_v3",
                )

                st.download_button(
                    "Завантажити параметри огороджень п.21",
                    data=dataframe_to_csv_bytes(
                        heat_loss_envelope21
                    ),
                    file_name=(
                        "stage21_envelope_heat_transfer.csv"
                    ),
                    mime="text/csv",
                    width="stretch",
                    key="download_stage21_envelope_v3",
                )


# ---------------------------------------------------------------------
# 17. РЕКУПЕРАЦІЯ ТЕПЛОТИ ВІДПРАЦЬОВАНОГО СУШИЛЬНОГО АГЕНТА
# ---------------------------------------------------------------------
with tab_recuperation:
    st.subheader(
        "Етап 22. Потенціал рекуперації теплоти відпрацьованого повітря"
    )

    st.info(
        "На цьому етапі не задається довільна ефективність "
        "рекуператора. Спочатку визначається теоретично доступна "
        "відчутна теплота відпрацьованого сушильного агента та "
        "виконується параметричний аналіз для ε = 0…1."
    )

    st.markdown(
        "#### Вхідні потоки"
    )

    st.latex(
        r"T_{\mathrm{гар,вх}}(t)"
        r"=T_{\mathrm{вих,кам}}(t)"
    )

    st.latex(
        r"T_{\mathrm{хол,вх}}(t)"
        r"=T_{\mathrm{зовн}}(t)"
    )

    st.latex(
        r"C_{\mathrm{гар}}(t)"
        r"=\dot m_{\mathrm{с.п.}}"
        r"\left(1.006+1.86d_{\mathrm{гар}}\right)"
    )

    st.latex(
        r"C_{\mathrm{хол}}(t)"
        r"=\dot m_{\mathrm{с.п.}}"
        r"\left(1.006+1.86d_{\mathrm{хол}}\right)"
    )

    st.latex(
        r"\dot Q_{\max}(t)"
        r"=C_{\min}(t)"
        r"\max\left["
        r"T_{\mathrm{гар,вх}}(t)"
        r"-T_{\mathrm{хол,вх}}(t),0"
        r"\right]"
    )

    st.write(
        "Для конкретного рекуператора після визначення його "
        "ефективності буде:"
    )

    st.latex(
        r"\dot Q_{\mathrm{рек}}(t)"
        r"=\varepsilon_{\mathrm{рек}}"
        r"\dot Q_{\max}(t)"
    )

    st.warning(
        "Поточна постановка відповідає сухому відчутному "
        "повітря-повітря рекуператору. Якщо гарячий потік "
        "охолоджується до точки роси, проста суха модель перестає "
        "бути достатньою. Програма показує таку ситуацію лише як "
        "діагностичний ризик конденсації за температурою середнього "
        "гарячого потоку; локальна температура стінки теплообмінника "
        "поки не моделюється."
    )

    if "coupled_drying_profile" not in st.session_state:
        st.warning(
            "Спочатку виконайте спільний розрахунок сушіння."
        )

    else:
        try:
            (
                recuperation_profile22,
                recuperation_scenarios22,
                recuperation_summary22,
            ) = calculate_recuperation_potential_stage22(
                coupled_profile=st.session_state[
                    "coupled_drying_profile"
                ],
                step_minutes=int(
                    model_step_minutes
                ),
            )

        except Exception as exc:
            st.warning(str(exc))

        else:
            st.session_state[
                "recuperation_profile"
            ] = recuperation_profile22

            st.session_state[
                "recuperation_scenarios"
            ] = recuperation_scenarios22

            st.session_state[
                "recuperation_summary"
            ] = recuperation_summary22

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Tгар,вх, сер",
                (
                    f"{recuperation_summary22['mean_hot_inlet_temp_c']:.2f} "
                    "°C"
                ),
            )

            c2.metric(
                "Tхол,вх, сер",
                (
                    f"{recuperation_summary22['mean_cold_inlet_temp_c']:.2f} "
                    "°C"
                ),
            )

            c3.metric(
                "Точка роси гарячого потоку",
                (
                    f"{recuperation_summary22['mean_hot_inlet_dew_point_c']:.2f} "
                    "°C"
                ),
            )

            c4.metric(
                "Пікова Q̇max",
                (
                    f"{recuperation_summary22['peak_qmax_sensible_kw']:.3f} "
                    "кВт"
                ),
            )

            c5, c6, c7 = st.columns(3)

            c5.metric(
                "Середня Q̇max",
                (
                    f"{recuperation_summary22['mean_qmax_sensible_kw']:.3f} "
                    "кВт"
                ),
            )

            c6.metric(
                "Теоретична Qmax за цикл",
                (
                    f"{recuperation_summary22['qmax_sensible_energy_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            c7.metric(
                "Макс. компенсація нагрівання повітря",
                (
                    f"{recuperation_summary22['maximum_theoretical_heating_offset_pct']:.1f} "
                    "%"
                ),
            )

            st.markdown(
                "#### Температури потоків і точка роси"
            )

            st.line_chart(
                recuperation_profile22[
                    [
                        "recup_hot_in_temp_c",
                        "recup_cold_in_temp_c",
                        "recup_hot_in_dew_point_c",
                    ]
                ].rename(
                    columns={
                        "recup_hot_in_temp_c": (
                            "Tгар,вх, °C"
                        ),
                        "recup_cold_in_temp_c": (
                            "Tхол,вх, °C"
                        ),
                        "recup_hot_in_dew_point_c": (
                            "Tdp,гар, °C"
                        ),
                    }
                ),
                width="stretch",
            )

            st.markdown(
                "#### Теоретично доступна потужність рекуперації"
            )

            st.line_chart(
                recuperation_profile22[
                    [
                        "recup_qmax_sensible_kw",
                        "recup_baseline_air_heating_kw",
                    ]
                ].rename(
                    columns={
                        "recup_qmax_sensible_kw": (
                            "Q̇max, кВт"
                        ),
                        "recup_baseline_air_heating_kw": (
                            "Q̇нагр,без рекуп., кВт"
                        ),
                    }
                ),
                width="stretch",
            )

            st.markdown(
                "#### Параметричний аналіз за ефективністю рекуператора"
            )

            scenarios_display22 = (
                recuperation_scenarios22.copy()
            )

            scenarios_display22["effectiveness"] = (
                scenarios_display22[
                    "effectiveness"
                ]
                * 100.0
            )

            scenarios_display22 = (
                scenarios_display22.rename(
                    columns={
                        "effectiveness": (
                            "εрек, %"
                        ),
                        "recovered_energy_kwh": (
                            "Qрек, кВт·год"
                        ),
                        "residual_air_heating_kwh": (
                            "Qдогр, кВт·год"
                        ),
                        "air_heating_reduction_pct": (
                            "Зменшення нагрівання, %"
                        ),
                        "mean_cold_outlet_temp_c": (
                            "Tхол,вих,сер, °C"
                        ),
                        "mean_hot_outlet_temp_c": (
                            "Tгар,вих,сер, °C"
                        ),
                        "bulk_dewpoint_risk_intervals": (
                            "Інтервали з діагн. ризиком конденсації"
                        ),
                    }
                )
            )

            st.dataframe(
                scenarios_display22,
                width="stretch",
                hide_index=True,
            )

            eps_limit22 = recuperation_summary22[
                "minimum_bulk_dewpoint_effectiveness_limit"
            ]

            if pd.notna(eps_limit22):
                st.write(
                    "Найменше значення ефективності, при якому "
                    "за поточною **діагностикою середнього гарячого "
                    "потоку** хоча б в одному активному інтервалі "
                    "його розрахункова температура на виході досягає "
                    "точки роси:"
                )

                st.metric(
                    "ε за bulk-dewpoint діагностикою",
                    f"{eps_limit22 * 100.0:.1f} %",
                )

                st.caption(
                    "Це не є допустимою паспортною межею "
                    "рекуператора і не гарантує відсутності "
                    "конденсації на стінці. Для точної перевірки "
                    "потрібні геометрія теплообмінника та локальні "
                    "температури його поверхні."
                )

            st.download_button(
                "Завантажити часовий ряд п.22",
                data=dataframe_to_csv_bytes(
                    recuperation_profile22
                ),
                file_name=(
                    "stage22_recuperation_potential.csv"
                ),
                mime="text/csv",
                width="stretch",
                key="download_stage22_profile",
            )

            st.download_button(
                "Завантажити сценарії ефективності п.22",
                data=dataframe_to_csv_bytes(
                    recuperation_scenarios22
                ),
                file_name=(
                    "stage22_recuperation_effectiveness_scenarios.csv"
                ),
                mime="text/csv",
                width="stretch",
                key="download_stage22_scenarios",
            )


# ---------------------------------------------------------------------
# 18. ЗВЕДЕНИЙ НЕСТАЦІОНАРНИЙ ТЕПЛОВИЙ БАЛАНС
# ---------------------------------------------------------------------
with tab_heat_balance:
    st.subheader(
        "Етап 23. Зведений нестаціонарний тепловий баланс"
    )

    st.info(
        "Мета п.23 — звести вже розраховані складові в одну "
        "енергетичну схему та не допустити подвійного обліку. "
        "На цьому етапі програма не вигадує часові профілі "
        "нагрівання продукту або конструкцій, яких у моделі ще немає."
    )

    st.markdown(
        "#### Динамічно замкнена частина балансу"
    )

    st.latex(
        r"\dot Q_{\mathrm{пов,після\,рек}}(t,\varepsilon)"
        r"=\max\left["
        r"\dot Q_{\mathrm{пов}}(t)"
        r"-\varepsilon_{\mathrm{рек}}\dot Q_{\max,\mathrm{рек}}(t),"
        r"0\right]"
    )

    st.latex(
        r"\dot Q_{\mathrm{відом}}(t,\varepsilon)"
        r"="
        r"\dot Q_{\mathrm{пов,після\,рек}}(t,\varepsilon)"
        r"+\dot Q_{\mathrm{втр}}(t)"
    )

    st.write(
        "Тут **не додається окремо** теплота фазового переходу "
        "з п.18. У зональній моделі п.15 зволоження повітря "
        "супроводжується його охолодженням при практично сталій "
        "ентальпії, тому повторне додавання Qвип завищило б "
        "зовнішню теплову потребу."
    )

    st.latex(
        r"h_{\mathrm{вих,кам}}(t)"
        r"\approx h_{\mathrm{вх,кам}}(t)"
    )

    st.markdown(
        "#### Що ще не можна довільно включити в часовий баланс"
    )

    st.write(
        "П.19 дає інтегральну проєктну енергію нагрівання продукту, "
        "але поки не визначає Q̇прод(t). П.20 дає теплову ємність "
        "конструкції Cконстр, але без Tконстр(t) неможливо отримати "
        "Q̇констр(t). Тому ці складові показуються окремо."
    )

    st.latex(
        r"\dot Q_{\mathrm{констр}}(t)"
        r"=C_{\mathrm{констр}}"
        r"\frac{dT_{\mathrm{констр}}}{dt}"
    )

    prerequisites23 = [
        "air_heating_profile",
        "heat_loss_profile",
        "recuperation_profile",
    ]

    missing23 = [
        key
        for key in prerequisites23
        if key not in st.session_state
    ]

    if missing23:
        st.warning(
            "Для п.23 спочатку мають бути сформовані п.17, п.21 "
            "і п.22. Відсутні: "
            + ", ".join(missing23)
        )

    else:
        try:
            (
                heat_balance_profile23,
                heat_balance_scenarios23,
                heat_balance_ledger23,
                heat_balance_summary23,
            ) = calculate_unified_heat_balance_stage23(
                air_heating_profile=st.session_state[
                    "air_heating_profile"
                ],
                heat_loss_profile=st.session_state[
                    "heat_loss_profile"
                ],
                recuperation_profile=st.session_state[
                    "recuperation_profile"
                ],
                product_summary=st.session_state.get(
                    "product_heating_summary"
                ),
                evaporation_summary=st.session_state.get(
                    "evaporation_energy_summary"
                ),
                construction_summary=st.session_state.get(
                    "construction_thermal_capacity_summary"
                ),
                step_minutes=int(
                    model_step_minutes
                ),
            )

        except Exception as exc:
            st.warning(str(exc))

        else:
            st.session_state[
                "heat_balance_profile"
            ] = heat_balance_profile23

            st.session_state[
                "heat_balance_scenarios"
            ] = heat_balance_scenarios23

            st.session_state[
                "heat_balance_ledger"
            ] = heat_balance_ledger23

            st.session_state[
                "heat_balance_summary"
            ] = heat_balance_summary23

            st.markdown(
                "#### Контроль узгодженості попередніх етапів"
            )

            k1, k2 = st.columns(2)

            k1.metric(
                "Макс. різниця п.17 ↔ базова потреба п.22",
                (
                    f"{heat_balance_summary23['max_stage17_stage22_baseline_error_kw']:.6f} "
                    "кВт"
                ),
            )

            k2.metric(
                "Макс. |ΔQ̇ентальпії| через камеру",
                (
                    f"{heat_balance_summary23['max_chamber_air_enthalpy_balance_error_kw']:.6f} "
                    "кВт"
                ),
            )

            st.caption(
                "Перша перевірка контролює, що п.22 використовує ту "
                "саму базову потребу нагрівання повітря, що й п.17. "
                "Друга показує, наскільки близько зональна модель п.15 "
                "дотримується сталої ентальпії сушильного агента."
            )

            st.markdown(
                "#### Енергетичний реєстр"
            )

            ledger_display23 = (
                heat_balance_ledger23.copy().rename(
                    columns={
                        "component": "Складова",
                        "role": "Роль у п.23",
                        "energy_kwh": "Енергія, кВт·год",
                        "time_resolved": "Часовий профіль",
                    }
                )
            )

            st.dataframe(
                ledger_display23,
                width="stretch",
                hide_index=True,
            )

            s1, s2, s3, s4 = st.columns(4)

            s1.metric(
                "Нагрівання повітря без рекуперації",
                (
                    f"{heat_balance_summary23['air_heating_no_rec_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            s2.metric(
                "Трансмісійні втрати",
                (
                    f"{heat_balance_summary23['transmission_loss_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            s3.metric(
                "Відомий динамічний баланс, ε=0",
                (
                    f"{heat_balance_summary23['known_dynamic_no_rec_kwh']:.3f} "
                    "кВт·год"
                ),
            )

            product23 = heat_balance_summary23[
                "product_design_heat_kwh"
            ]

            if pd.notna(product23):
                product23_text = (
                    f"{product23:.3f} кВт·год"
                )
            else:
                product23_text = "не визначено"

            s4.metric(
                "Проєктна теплота нагрівання продукту",
                product23_text,
            )

            st.markdown(
                "#### Параметричний баланс за ефективністю рекуперації"
            )

            scenario_display23 = (
                heat_balance_scenarios23.copy()
            )

            scenario_display23["effectiveness"] = (
                scenario_display23[
                    "effectiveness"
                ]
                * 100.0
            )

            scenario_display23 = (
                scenario_display23.rename(
                    columns={
                        "effectiveness": "εрек, %",
                        "recovered_sensible_energy_kwh": (
                            "Qрек, кВт·год"
                        ),
                        "fresh_air_heating_after_rec_kwh": (
                            "Qпов після рек., кВт·год"
                        ),
                        "transmission_loss_kwh": (
                            "Qвтр, кВт·год"
                        ),
                        "known_dynamic_source_energy_kwh": (
                            "Eвідом,дин, кВт·год"
                        ),
                        "product_design_heat_kwh": (
                            "Qпрод,проєкт, кВт·год"
                        ),
                        "known_cycle_energy_plus_product_kwh": (
                            "Eвідом + Qпрод, кВт·год"
                        ),
                        "construction_storage_status": (
                            "Конструкція"
                        ),
                        "source_reduction_vs_no_rec_pct": (
                            "Зменшення Eвідом, %"
                        ),
                    }
                )
            )

            st.dataframe(
                scenario_display23,
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Стовпець «Eвідом + Qпрод» ще НЕ є остаточною "
                "тепловою потребою установки, оскільки відсутній "
                "динамічний член накопичення теплоти конструкцією."
            )

            st.markdown(
                "#### Динамічна теплова потужність для трьох граничних сценаріїв"
            )

            st.caption(
                "ε = 0 %, 50 % і 100 % наведені лише для порівняння; "
                "це не вибір проєктної ефективності рекуператора."
            )

            st.line_chart(
                heat_balance_profile23[
                    [
                        "balance_known_power_eps_0_kw",
                        "balance_known_power_eps_50_kw",
                        "balance_known_power_eps_100_kw",
                    ]
                ].rename(
                    columns={
                        "balance_known_power_eps_0_kw": (
                            "Q̇відом, ε=0 %, кВт"
                        ),
                        "balance_known_power_eps_50_kw": (
                            "Q̇відом, ε=50 %, кВт"
                        ),
                        "balance_known_power_eps_100_kw": (
                            "Q̇відом, ε=100 %, кВт"
                        ),
                    }
                ),
                width="stretch",
            )

            st.markdown(
                "#### Невизначений член конструкції"
            )

            construction_capacity23 = (
                heat_balance_summary23[
                    "construction_capacity_kj_k"
                ]
            )

            if pd.notna(
                construction_capacity23
            ):
                st.metric(
                    "Cконстр",
                    (
                        f"{construction_capacity23:.2f} "
                        "кДж/К"
                    ),
                )

                st.write(
                    "Це значення вже відоме з п.20, але перетворювати "
                    "його на кВт·год без розрахованого Tконстр(t) "
                    "програма не буде."
                )

            evaporation23 = (
                heat_balance_summary23[
                    "evaporation_diagnostic_kwh"
                ]
            )

            if pd.notna(evaporation23):
                st.metric(
                    "Qвип з п.18 — лише діагностична характеристика",
                    (
                        f"{evaporation23:.3f} "
                        "кВт·год"
                    ),
                )

            st.warning(
                "П.23 навмисно не називає поточну суму «остаточною "
                "тепловою потребою». Для повного нестаціонарного "
                "балансу ще потрібні температурна динаміка продукту "
                "та конструкції. Після їх введення баланс можна буде "
                "замкнути без емпіричного розподілу енергії по часу."
            )

            st.download_button(
                "Завантажити часовий ряд п.23",
                data=dataframe_to_csv_bytes(
                    heat_balance_profile23
                ),
                file_name=(
                    "stage23_unified_heat_balance_profile.csv"
                ),
                mime="text/csv",
                width="stretch",
                key="download_stage23_profile",
            )

            st.download_button(
                "Завантажити сценарії балансу п.23",
                data=dataframe_to_csv_bytes(
                    heat_balance_scenarios23
                ),
                file_name=(
                    "stage23_unified_heat_balance_scenarios.csv"
                ),
                mime="text/csv",
                width="stretch",
                key="download_stage23_scenarios",
            )

            st.info(
                "Наступний логічний крок — визначити, як замкнути "
                "динамічні члени нагрівання продукту та конструкції, "
                "після чого перейти до потрібної теплової потужності "
                "джерела і моделі сонячного колектора."
            )

