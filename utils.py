import json

def clean_user_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    return " ".join(raw_text.split())

def normalize_metadata(meta: dict) -> dict:
    normalized = {}
    for k, v in meta.items():
        if isinstance(v, list):
            normalized[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
            normalized[k] = json.dumps(v)
        else:
            normalized[k] = v
    return normalized
