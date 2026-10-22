"""Clinical Simulation Console — Flask backend and waveform stream."""
from __future__ import annotations

import json
import math
import os
import threading
import time

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "pomc_model.joblib")
TARGET_STREAMING_FS = 128

WAVEFORM_FILES = {
    "ecg": "NOM_ECG_ELEC_POTL_IIWaveExport.csv",
    "ppg": "NOM_PLETHWaveExport.csv",
    "abp": "NOM_PRESS_BLD_ART_ABPWaveExport.csv",
    "resp": "NOM_RESPWaveExport.csv",
}

PATIENTS = {
    "patient-a": {
        "name": "Patient A",
        "procedure": "Abdominal surgery",
        "age": "Adult",
        "sex": "Not specified",
        "issue": "Postoperative hypovolemia",
        "scenario": [
            {"name": "Initial", "hr": 92, "spo2": 99, "sys": 102, "dia": 68, "rr": 16, "temp": 37.1, "pain": 7, "cue": "Assess severe pain and verify epidural function before additional opioids."},
            {"name": "Active case", "hr": 94, "spo2": 97, "sys": 90, "dia": 65, "rr": 14, "temp": 36.9, "pain": 5, "cue": "Recognize hypotension and lethargy; inspect for bleeding and notify the team."},
            {"name": "Resolution", "hr": 90, "spo2": 99, "sys": 105, "dia": 70, "rr": 16, "temp": 37.0, "pain": 3, "cue": "Reassess after the 500 mL crystalloid bolus and communicate the response."},
        ],
    },
    "patient-b": {
        "name": "Patient B",
        "procedure": "Thoracic surgery",
        "age": "Adult",
        "sex": "Not specified",
        "issue": "Hypoxia-driven agitation",
        "scenario": [
            {"name": "Initial", "hr": 104, "spo2": 91, "sys": 138, "dia": 82, "rr": 24, "temp": 36.8, "pain": None, "cue": "Prioritize oxygenation, respiratory assessment, and calm reorientation."},
            {"name": "Active case", "hr": 118, "spo2": 85, "sys": 148, "dia": 90, "rr": 28, "temp": 36.7, "pain": None, "cue": "Trace the chest tube: it is clamped between the dressing and collection chamber."},
            {"name": "Resolution", "hr": 92, "spo2": 96, "sys": 128, "dia": 78, "rr": 18, "temp": 36.8, "pain": 2, "cue": "Confirm improved oxygenation, lung expansion, chest-tube function, and mentation."},
        ],
    },
    "patient-c": {
        "name": "Patient C",
        "procedure": "Abdominal surgery",
        "age": "Adult",
        "sex": "Not specified",
        "issue": "Self-resolving SVT",
        "scenario": [
            {"name": "Initial", "hr": 88, "spo2": 97, "sys": 132, "dia": 78, "rr": 18, "temp": 36.9, "pain": 4, "cue": "Assess referred shoulder pressure and rule out cardiac or respiratory causes."},
            {"name": "SVT event", "hr": 168, "spo2": 97, "sys": 128, "dia": 76, "rr": 20, "temp": 36.9, "pain": 4, "cue": "Capture the rhythm, assess stability, obtain ECG/labs, and notify the team."},
            {"name": "Resolution", "hr": 86, "spo2": 98, "sys": 130, "dia": 76, "rr": 16, "temp": 36.8, "pain": 2, "cue": "Correct the documented electrolyte abnormality, continue telemetry, and reassess."},
        ],
    },
}


def synthetic_waveforms(points: int = 4096) -> dict[str, list[float]]:
    """Generate deterministic normalized fallback waveforms."""
    result = {"ecg": [], "ppg": [], "abp": [], "resp": []}
    for i in range(points):
        phase = (i % 128) / 128
        ecg = 0.34 + 0.02 * math.sin(phase * math.tau)
        ecg += 0.12 * math.exp(-((phase - 0.20) / 0.035) ** 2)
        ecg -= 0.18 * math.exp(-((phase - 0.250) / 0.022) ** 2)
        # Keep the QRS wide enough to remain visible at high heart rates on a
        # 128 Hz display; an overly narrow peak can be skipped during resampling.
        ecg += 0.62 * math.exp(-((phase - 0.285) / 0.026) ** 2)
        ecg -= 0.22 * math.exp(-((phase - 0.325) / 0.023) ** 2)
        ecg += 0.16 * math.exp(-((phase - 0.54) / 0.09) ** 2)
        ppg = 0.18 + (phase / 0.18) ** 2 if phase < 0.18 else 1.18 * math.exp(-(phase - 0.18) * 3.3)
        abp = 0.22 + 0.78 * (phase / 0.13) if phase < 0.13 else 0.22 + 0.78 * math.exp(-(phase - 0.13) * 3.5)
        abp += 0.06 * math.exp(-((phase - 0.46) / 0.035) ** 2)
        resp_phase = (i % 480) / 480
        resp = 0.5 + 0.42 * math.sin(resp_phase * math.tau - math.pi / 2)
        result["ecg"].append(float(max(0, min(1, ecg))))
        result["ppg"].append(float(max(0, min(1, ppg))))
        result["abp"].append(float(max(0, min(1, abp))))
        result["resp"].append(float(max(0, min(1, resp))))
    return result


def compile_waveforms() -> dict[str, list[float]]:
    registry = synthetic_waveforms()
    for key, filename in WAVEFORM_FILES.items():
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            # Philips exports are headerless: timestamp, sequence, export time,
            # signal value, and a trailing empty column.
            df = pd.read_csv(path, header=None, usecols=[0, 3], names=["timestamp", "value"])
            values = pd.to_numeric(df["value"], errors="coerce")
            times = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S.%f", errors="coerce")
            valid = values.notna() & times.notna()
            signal = values[valid].to_numpy(dtype=float)
            signal_times = times[valid]
            if len(signal) < 2:
                continue

            if key == "ecg":
                # This Philips ECG export contains one isolated 2.02-valued
                # needle per cardiac cycle in addition to the physiologic QRS.
                # It is an export/marker artifact, not a second heartbeat.
                # Replace only those extreme one-sample needles by linear
                # interpolation before resampling, preserving the real QRS.
                physiologic_top = float(np.quantile(signal, 0.997))
                artifact_mask = signal > max(1.0, physiologic_top * 1.5)
                if artifact_mask.any() and (~artifact_mask).sum() >= 2:
                    sample_axis = np.arange(len(signal))
                    signal[artifact_mask] = np.interp(
                        sample_axis[artifact_mask],
                        sample_axis[~artifact_mask],
                        signal[~artifact_mask],
                    )
                    print(f"Removed {int(artifact_mask.sum())} ECG export-marker samples")

            elapsed = (signal_times.iloc[-1] - signal_times.iloc[0]).total_seconds()
            source_fs = (len(signal) - 1) / elapsed if elapsed > 0 else TARGET_STREAMING_FS
            output_count = max(2, round(len(signal) * TARGET_STREAMING_FS / source_fs))
            source_axis = np.linspace(0.0, 1.0, len(signal), endpoint=True)
            output_axis = np.linspace(0.0, 1.0, output_count, endpoint=True)
            signal = np.interp(output_axis, source_axis, signal)
            low, high = float(signal.min()), float(signal.max())
            registry[key] = ((signal - low) / (high - low)).tolist() if high != low else [0.5] * 2000
            print(f"Loaded {key.upper()} from CSV: {source_fs:.1f} Hz -> {TARGET_STREAMING_FS} Hz")
        except Exception as exc:
            print(f"Waveform load failed for {key}: {exc}")
    return registry


WAVEFORMS = compile_waveforms()


def estimate_cycle_rate(signal: list[float], kind: str) -> float:
    """Estimate the recording's native cycles/minute for calibrated playback."""
    values = np.asarray(signal, dtype=float)
    smoothing_seconds = 0.25 if kind == "resp" else 0.03
    window = max(1, round(TARGET_STREAMING_FS * smoothing_seconds))
    smooth = np.convolve(values, np.ones(window) / window, mode="same")
    threshold = float(np.quantile(smooth, 0.70))
    candidates = np.where(
        (smooth[1:-1] > smooth[:-2])
        & (smooth[1:-1] >= smooth[2:])
        & (smooth[1:-1] > threshold)
    )[0] + 1
    minimum_gap = round(TARGET_STREAMING_FS * (1.2 if kind == "resp" else 0.35))
    peaks: list[int] = []
    for candidate in candidates:
        if not peaks or candidate - peaks[-1] >= minimum_gap:
            peaks.append(int(candidate))
        elif smooth[candidate] > smooth[peaks[-1]]:
            peaks[-1] = int(candidate)
    if len(peaks) < 3:
        return 16.0 if kind == "resp" else 60.0
    return float(60.0 * TARGET_STREAMING_FS / np.median(np.diff(peaks)))


SOURCE_PULSE_RATE = estimate_cycle_rate(WAVEFORMS["ecg"], "pulse")
SOURCE_RESP_RATE = estimate_cycle_rate(WAVEFORMS["resp"], "resp")
print(f"Calibrated source rates: pulse {SOURCE_PULSE_RATE:.1f}/min, respiration {SOURCE_RESP_RATE:.1f}/min")
MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
MONITOR_LOCK = threading.Lock()
MONITOR_TARGET = {"hr": 104.0, "spo2": 91.0, "sys": 138.0, "dia": 82.0, "rr": 24.0}


def as_float(value, default=0.0):
    try:
        return float(value) if value not in (None, "", "nan") else default
    except (TypeError, ValueError):
        return default


def interpolate_cyclic(signal: list[float], cursor: float) -> float:
    """Sample a looping recording without nearest-index beat distortion."""
    length = len(signal)
    position = cursor % length
    left = int(position)
    fraction = position - left
    right = (left + 1) % length
    return float(signal[left] + (signal[right] - signal[left]) * fraction)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index4.html")


@app.get("/api/patients")
def patients():
    return jsonify(PATIENTS)


@app.post("/predict/single")
def predict_single():
    try:
        started = time.perf_counter()
        data = request.get_json(force=True)
        if MODEL is not None:
            procedure = str(data.get("proc", "elective")).lower().strip()
            comorbidities = data.get("comorbidities", [])
            features = [
                as_float(data.get("age")), 1.0 if str(data.get("sex", "")).upper() in ("M", "MALE", "1") else 0.0,
                as_float(data.get("bmi")), as_float(data.get("asa"), 1.0), as_float(data.get("hgb")),
                as_float(data.get("albumin")), as_float(data.get("creatinine")), as_float(data.get("wbc")),
                as_float(data.get("sodium")), as_float(data.get("inr")),
                float(procedure == "elective"), float(procedure == "urgent"), float(procedure == "emergent"),
                as_float(data.get("ebl")), as_float(data.get("op_duration")), as_float(data.get("wound_class"), 1.0),
                *[float(name in comorbidities) for name in ("dm", "htn", "chf", "copd", "ckd", "cancer", "smoker", "steroid")],
            ]
            probability = float(MODEL.predict_proba(np.array(features).reshape(1, -1))[0][1])
        else:
            probability = min(max((as_float(data.get("age"), 50) - 18) / 80, 0), 1) * 0.20
        return jsonify({
            "probability": round(probability, 4),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "demo": MODEL is None,
            "model_version": "demo" if MODEL is None else "Outcome model",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/stream/waveforms")
def stream_waveforms():
    @stream_with_context
    def generate():
        pulse_cursor = 0.0
        resp_cursor = 0.0
        tick = 0
        with MONITOR_LOCK:
            current = dict(MONITOR_TARGET)
        while True:
            with MONITOR_LOCK:
                target = dict(MONITOR_TARGET)

            # Ease rather than jump. The stream and horizontal cursor never reset.
            # At 128 Hz, 0.05 gives a visible transition in roughly 0.4 seconds
            # while preserving phase and stream continuity.
            for key in current:
                current[key] += (target[key] - current[key]) * 0.05

            # The CSV recordings provide the morphology. Scenario vitals transform
            # playback rate and amplitude so the traces agree with the monitor.
            ecg = interpolate_cyclic(WAVEFORMS["ecg"], pulse_cursor)
            ppg_raw = interpolate_cyclic(WAVEFORMS["ppg"], pulse_cursor)
            abp_raw = interpolate_cyclic(WAVEFORMS["abp"], pulse_cursor)
            resp_raw = interpolate_cyclic(WAVEFORMS["resp"], resp_cursor)

            # SpO2 is calculated from the red/infrared optical ratio, not from
            # the visible pleth amplitude. Preserve the recorded PPG morphology;
            # its pulse timing follows HR while SpO2 remains the numeric reading.
            ppg = ppg_raw
            arterial_mmhg = current["dia"] + abp_raw * (current["sys"] - current["dia"])
            abp = max(0.0, min(1.0, (arterial_mmhg - 40.0) / 140.0))
            payload = {
                "ecg": ecg,
                "ppg": ppg,
                "abp": abp,
                "resp": resp_raw,
                "hr": current["hr"],
                "spo2": current["spo2"],
                "sys": current["sys"],
                "dia": current["dia"],
                "rr": current["rr"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            # Calibrate playback against the cycle rates measured in the actual
            # CSV recordings. Both cursors remain continuous across scenarios.
            pulse_cursor += max(0.20, current["hr"] / SOURCE_PULSE_RATE)
            resp_cursor += max(0.20, current["rr"] / SOURCE_RESP_RATE)
            tick += 1
            time.sleep(1 / TARGET_STREAMING_FS)

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/monitor-state")
def monitor_state():
    data = request.get_json(force=True)
    updated = {
        "hr": max(20.0, min(240.0, as_float(data.get("hr"), 80))),
        "spo2": max(50.0, min(100.0, as_float(data.get("spo2"), 98))),
        "sys": max(50.0, min(260.0, as_float(data.get("sys"), 120))),
        "dia": max(20.0, min(180.0, as_float(data.get("dia"), 80))),
        "rr": max(4.0, min(60.0, as_float(data.get("rr"), 16))),
    }
    if updated["dia"] >= updated["sys"]:
        return jsonify({"error": "Diastolic pressure must be below systolic pressure"}), 400
    with MONITOR_LOCK:
        MONITOR_TARGET.update(updated)
    return jsonify({"status": "transitioning", "target": updated})


@app.get("/health")
def health():
    waveform_sources = {
        key: ("csv" if os.path.exists(os.path.join(BASE_DIR, filename)) else "synthetic")
        for key, filename in WAVEFORM_FILES.items()
    }
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "waveform_sources": waveform_sources,
        "source_rates_per_min": {"pulse": round(SOURCE_PULSE_RATE, 2), "resp": round(SOURCE_RESP_RATE, 2)},
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
