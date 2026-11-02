from __future__ import annotations

import math
import os
from pathlib import Path

from functools import lru_cache

import joblib
import pandas as pd

from src.config import (
    FEATURES,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    FLIGHTS_PATH,
    TARGET,
)

from src.historical_features import (
    MODEL_FEATURES,
    prepare_historical_reference,
    prepare_prediction_features,
)


CANDIDATE_MODEL_PATH = (
    MODEL_DIR / "flight_delay_model_cyclic_candidate.joblib"
)

HISTORICAL_CANDIDATE_MODEL_PATH = (
    MODEL_DIR / "flight_delay_model_historical_candidate.joblib"
)


@lru_cache(maxsize=1)
def _load_historical_reference_cached(
    path: str,
    modified_ns: int,
):
    """
    Read and normalize historical data once per dataset version.

    modified_ns is included in the cache key so replacing the
    Parquet file results in a fresh cached reference.
    """

    columns = list(
        dict.fromkeys([*FEATURES, TARGET, "FL_DATE"])
    )

    raw_history = pd.read_parquet(
        Path(path),
        columns=columns,
    )

    return prepare_historical_reference(raw_history)


def load_historical_reference():
    """Load the cached, normalized historical reference data."""

    path = Path(FLIGHTS_PATH)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset historis tidak ditemukan: {path}"
        )

    modified_ns = path.stat().st_mtime_ns

    return _load_historical_reference_cached(
        str(path.resolve()),
        modified_ns,
    )


def _validate_threshold(value):
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Threshold model tidak valid."
        ) from exc

    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError(
            "Threshold model harus berada antara 0 dan 1."
        )

    return threshold


@lru_cache(maxsize=1)
def load_model_bundle():
    """
    Load a model and its classification threshold.

    Supported MODEL_SOURCE values:

    mlflow:
        Load the production champion from MLflow.
        Default threshold = 0.50.

    joblib/local/file:
        Load the existing local production model.
        Default threshold = 0.50.

    candidate:
        Load the separate Logistic Regression Cyclic candidate
        and its calibrated threshold.
    """

    model_source = (
        os.getenv(
            "MODEL_SOURCE",
            "mlflow",
        )
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # Historical candidate model
    # --------------------------------------------------------

    if model_source in {
        "historical_candidate",
        "cyclical_historical_candidate",
    }:

        if not HISTORICAL_CANDIDATE_MODEL_PATH.exists():
            raise RuntimeError(
                "Model historical candidate tidak ditemukan: "
                f"{HISTORICAL_CANDIDATE_MODEL_PATH}"
            )

        artifact = joblib.load(
            HISTORICAL_CANDIDATE_MODEL_PATH
        )

        if not isinstance(artifact, dict):
            raise RuntimeError(
                "Format historical candidate tidak valid."
            )

        if "model" not in artifact or "threshold" not in artifact:
            raise RuntimeError(
                "Historical candidate tidak memiliki model atau threshold."
            )

        if list(artifact.get("feature_schema", [])) != list(MODEL_FEATURES):
            raise RuntimeError(
                "Feature schema historical candidate tidak cocok "
                "dengan historical_features.py."
            )

        model = artifact["model"]

        if not hasattr(model, "predict_proba"):
            raise RuntimeError(
                "Historical candidate tidak memiliki predict_proba."
            )

        model_schema = getattr(model, "feature_names_in_", None)

        if model_schema is None or list(model_schema) != list(MODEL_FEATURES):
            raise RuntimeError(
                "Feature schema internal model tidak cocok."
            )

        return {
            "model": model,
            "threshold": _validate_threshold(
                artifact["threshold"]
            ),
            "model_name": artifact.get(
                "model_name",
                "logistic_regression_cyclical_historical_rates",
            ),
            "model_source": "historical_candidate",
            "requires_historical_features": True,
        }

    # --------------------------------------------------------
    # Existing candidate model
    # --------------------------------------------------------

    if model_source in {
        "candidate",
        "joblib_candidate",
        "cyclic_candidate",
    }:

        if not CANDIDATE_MODEL_PATH.exists():
            raise RuntimeError(
                "Model kandidat tidak ditemukan: "
                f"{CANDIDATE_MODEL_PATH}"
            )

        artifact = joblib.load(
            CANDIDATE_MODEL_PATH
        )

        if not isinstance(artifact, dict):
            raise RuntimeError(
                "Format artefak model kandidat tidak valid."
            )

        if "model" not in artifact or "threshold" not in artifact:
            raise RuntimeError(
                "Artefak kandidat tidak memiliki model atau threshold."
            )

        model = artifact["model"]

        threshold = _validate_threshold(
            artifact["threshold"]
        )

        if not hasattr(model, "predict_proba"):
            raise RuntimeError(
                "Model kandidat tidak memiliki predict_proba."
            )

        return {
            "model": model,
            "threshold": threshold,
            "model_name": artifact.get(
                "model_name",
                "logistic_regression_cyclic",
            ),
            "model_source": "candidate",
        }

    # --------------------------------------------------------
    # Existing local production model
    # --------------------------------------------------------

    if model_source in {
        "joblib",
        "local",
        "file",
    }:

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model lokal tidak ditemukan: {MODEL_PATH}"
            )

        model = joblib.load(
            MODEL_PATH
        )

        return {
            "model": model,
            "threshold": 0.50,
            "model_name": MODEL_NAME,
            "model_source": "local_production",
        }

    # --------------------------------------------------------
    # MLflow production champion
    # --------------------------------------------------------

    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        MLFLOW_TRACKING_URI,
    )

    try:
        import mlflow.sklearn

        model = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}@champion"
        )

        return {
            "model": model,
            "threshold": 0.50,
            "model_name": MODEL_NAME,
            "model_source": "mlflow_champion",
        }

    except Exception as registry_error:

        if MODEL_PATH.exists():

            print(
                "Peringatan: model MLflow Registry belum bisa "
                f"dimuat ({registry_error}); memakai artifact lokal."
            )

            model = joblib.load(
                MODEL_PATH
            )

            return {
                "model": model,
                "threshold": 0.50,
                "model_name": MODEL_NAME,
                "model_source": "local_fallback",
            }

        raise RuntimeError(
            "Model belum tersedia. Bootstrap/training harus "
            "berhasil terlebih dahulu."
        ) from registry_error


def load_trained_model():
    """
    Backward-compatible model loader.

    Existing code that only needs the estimator
    can continue calling load_trained_model().
    """

    return load_model_bundle()["model"]


def _interpret_probability(
    predicted: int,
    probability: float | None,
    threshold: float = 0.50,
) -> str:
    """
    Explain the prediction using the model's actual threshold.

    The default remains 0.50 to preserve old behavior and tests.
    """

    if probability is None:
        return (
            "Probabilitas tidak tersedia; label model tidak boleh "
            "ditafsirkan sebagai kepastian kondisi penerbangan."
        )

    threshold = _validate_threshold(
        threshold
    )

    threshold_label = f"{threshold:.0%}"

    if abs(probability - threshold) <= 0.05:
        return (
            f"Probabilitas berada dekat ambang klasifikasi "
            f"{threshold_label}. Label kelas menunjukkan keputusan "
            "model pada ambang tersebut dan bukan bukti bahwa "
            "penerbangan akan tepat waktu."
        )

    if predicted == 1:
        return (
            f"Probabilitas berada pada atau di atas ambang klasifikasi "
            f"{threshold_label}; ini tetap estimasi historis, "
            "bukan kepastian delay."
        )

    return (
        f"Probabilitas berada di bawah ambang klasifikasi "
        f"{threshold_label}; ini bukan jaminan bahwa penerbangan "
        "akan tepat waktu."
    )


def predict_delay(
    flight_inputs: dict | None,
) -> dict:
    """
    Predict flight delay using the selected model and threshold.
    """

    required = FEATURES

    if not flight_inputs:
        return {
            "ok": False,
            "message": (
                "Detail penerbangan belum disimpan. "
                "Isi formulir pada sidebar terlebih dahulu."
            ),
        }

    missing = [
        name
        for name in required
        if (
            name not in flight_inputs
            or flight_inputs[name] in (None, "")
        )
    ]

    if missing:
        return {
            "ok": False,
            "message": (
                "Input belum lengkap: "
                + ", ".join(missing)
            ),
        }

    row = {
        name: flight_inputs[name]
        for name in required
    }

    # --------------------------------------------------------
    # Validate numeric features
    # --------------------------------------------------------

    numeric_fields = [
        "CRS_DEP_TIME",
        "MONTH",
        "DAY_OF_WEEK",
        "DISTANCE",
    ]

    for field in numeric_fields:

        try:
            row[field] = float(
                row[field]
            )

        except (TypeError, ValueError):
            return {
                "ok": False,
                "message": f"Nilai {field} tidak valid.",
            }

        if not math.isfinite(row[field]):
            return {
                "ok": False,
                "message": f"Nilai {field} harus berupa angka valid.",
            }

    # --------------------------------------------------------
    # Validate scheduled departure time
    # --------------------------------------------------------

    departure_time = row["CRS_DEP_TIME"]

    if (
        departure_time != int(departure_time)
        or departure_time < 0
        or departure_time > 2359
        or int(departure_time) % 100 >= 60
    ):
        return {
            "ok": False,
            "message": (
                "CRS_DEP_TIME tidak valid. "
                "Gunakan format HHMM, misalnya 1649 untuk 16:49."
            ),
        }

    # --------------------------------------------------------
    # Validate calendar features
    # --------------------------------------------------------

    if (
        row["MONTH"] != int(row["MONTH"])
        or not 1 <= row["MONTH"] <= 12
    ):
        return {
            "ok": False,
            "message": "MONTH harus berupa angka 1 sampai 12.",
        }

    if (
        row["DAY_OF_WEEK"] != int(row["DAY_OF_WEEK"])
        or not 1 <= row["DAY_OF_WEEK"] <= 7
    ):
        return {
            "ok": False,
            "message": "DAY_OF_WEEK harus berupa angka 1 sampai 7.",
        }

    if row["DISTANCE"] <= 0:
        return {
            "ok": False,
            "message": "DISTANCE harus lebih besar dari 0.",
        }

    # --------------------------------------------------------
    # Validate categorical features
    # --------------------------------------------------------

    for field in [
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
    ]:

        row[field] = (
            str(row[field])
            .strip()
            .upper()
        )

        if not row[field]:
            return {
                "ok": False,
                "message": f"Nilai {field} tidak boleh kosong.",
            }

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:
        bundle = load_model_bundle()

        model = bundle["model"]
        threshold = bundle["threshold"]

        if bundle.get("requires_historical_features", False):

            raw_prediction_date = flight_inputs.get("FL_DATE")

            if raw_prediction_date in (None, ""):
                raise ValueError(
                    "Model historical membutuhkan tanggal keberangkatan "
                    "(FL_DATE). Silakan simpan ulang detail penerbangan."
                )

            prediction_date = pd.to_datetime(
                raw_prediction_date,
                errors="coerce",
            )

            if pd.isna(prediction_date):
                raise ValueError(
                    "Tanggal keberangkatan (FL_DATE) tidak valid."
                )

            if getattr(prediction_date, "tzinfo", None) is not None:
                prediction_date = prediction_date.tz_localize(None)

            # Prevent inconsistent calendar inputs.
            if int(row["MONTH"]) != prediction_date.month:
                raise ValueError(
                    "MONTH tidak sesuai dengan FL_DATE."
                )

            expected_weekday = (
                (prediction_date.weekday() + 1) % 7
            ) + 1

            if int(row["DAY_OF_WEEK"]) != expected_weekday:
                raise ValueError(
                    "DAY_OF_WEEK tidak sesuai dengan FL_DATE. "
                    "Gunakan urutan BTS: Minggu=1 sampai Sabtu=7."
                )

            prediction_inputs = dict(row)
            prediction_inputs["FL_DATE"] = prediction_date

            historical_reference = load_historical_reference()

            frame = prepare_prediction_features(
                prediction_inputs,
                historical_reference,
            )

        else:
            # Original seven-feature prediction path remains unchanged.
            frame = pd.DataFrame(
                [row],
                columns=FEATURES,
            )

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                frame
            )[0]

            classes = list(
                model.classes_
            )

            if 1 in classes:
                probability = float(
                    probabilities[classes.index(1)]
                )

        # Apply the selected model threshold explicitly.
        # If probability is unavailable, use model.predict().
        if probability is not None:
            predicted = int(
                probability >= threshold
            )
        else:
            predicted = int(
                model.predict(frame)[0]
            )

        threshold_label = f"{threshold:.0%}"

        if predicted == 1:
            message = (
                "Model memperkirakan risiko keberangkatan "
                f"terlambat minimal 15 menit pada ambang "
                f"klasifikasi {threshold_label}."
            )
        else:
            message = (
                "Model tidak mengklasifikasikan penerbangan "
                f"ini sebagai delay minimal 15 menit pada "
                f"ambang klasifikasi {threshold_label}."
            )

        return {
            "ok": True,
            "predicted_delay_15": predicted,
            "probability": probability,
            "threshold": threshold,
            "interpretation": _interpret_probability(
                predicted,
                probability,
                threshold,
            ),
            "message": message,
            "model_name": bundle.get(
                "model_name",
                "unknown",
            ),
            "model_source": bundle.get(
                "model_source",
                "unknown",
            ),
            "features": row,
            "caveat": (
                "Estimasi berbasis pola historis, bukan status "
                "real-time dan bukan jaminan hasil aktual."
            ),
        }

    except Exception as exc:
        return {
            "ok": False,
            "message": f"Prediksi belum tersedia: {exc}",
        }