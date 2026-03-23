def get_sensor_data(asset_id: str):
    return f"{asset_id}: vibration high, pressure low"

def detect_anomaly(data: str):
    return "anomaly detected"

def map_failure(anomaly: str):
    return "cavitation"

def generate_work_order(asset_id: str, failure: str):
    return f"Work order created for {asset_id}: fix {failure}"