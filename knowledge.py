def failure_mode_plugin(symptoms: str):
    if "vibration" in symptoms:
        return ["cavitation", "bearing failure"]
    return ["unknown"]

def maintenance_policy_plugin(failure: str):
    if failure in ["cavitation", "bearing failure"]:
        return "create_work_order"
    return "monitor"