# skills.py

from tools import get_sensor_data, detect_anomaly, map_failure, generate_work_order

def root_cause_analysis(asset_id: str):
    data = get_sensor_data(asset_id)
    anomaly = detect_anomaly(data)
    failure = map_failure(anomaly)
    return {"failure": failure, "data": data}

def validate_failure(asset_id: str, failure: str):
    # simple placeholder
    return {"validated": True, "failure": failure}

def create_work_order(asset_id: str, failure: str):
    return generate_work_order(asset_id, failure)