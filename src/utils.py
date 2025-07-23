import json
from pathlib import Path
from typing import Dict, Any
from jsonschema import validate, ValidationError


def load_json_schema(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_output_json(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    try:
        validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        print(f"Schema validation error: {e.message}")
        return False
