"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: config_loader.py
Description: This file defines helper functions for loading simulator
configuration files from YAML. It provides a unified configuration loading
entry for simulation, cell, UE, RIC, and beamforming settings.
"""

from pathlib import Path
from typing import Any, Dict

import importlib


def _strip_simple_yaml_comment(line: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    out: list[str] = []
    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\" and in_double_quote:
            out.append(char)
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        if char == "#" and not in_single_quote and not in_double_quote:
            break
        out.append(char)
    return "".join(out).rstrip()


def _split_simple_yaml_items(value: str, *, separator: str = ",") -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    in_single_quote = False
    in_double_quote = False
    escaped = False
    for char in str(value):
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and in_double_quote:
            current.append(char)
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(char)
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(char)
            continue
        if not in_single_quote and not in_double_quote:
            if char in "[{(":
                depth += 1
            elif char in "]})":
                depth = max(0, depth - 1)
            elif char == separator and depth == 0:
                item = "".join(current).strip()
                if item != "":
                    items.append(item)
                current = []
                continue
        current.append(char)
    item = "".join(current).strip()
    if item != "":
        items.append(item)
    return items


def _split_simple_yaml_key_value(value: str) -> tuple[str, str] | None:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    depth = 0
    for index, char in enumerate(str(value)):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double_quote:
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if in_single_quote or in_double_quote:
            continue
        if char in "[{(":
            depth += 1
            continue
        if char in "]})":
            depth = max(0, depth - 1)
            continue
        if char == ":" and depth == 0:
            return str(value[:index]).strip(), str(value[index + 1 :]).strip()
    return None


def _coerce_simple_yaml_scalar(value: str) -> Any:
    text = str(value or "").strip()
    if text == "":
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return (
            text[1:-1]
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\\\", "\\")
        )
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner == "":
            return []
        return [_coerce_simple_yaml_scalar(item) for item in _split_simple_yaml_items(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if inner == "":
            return {}
        out: dict[str, Any] = {}
        for item in _split_simple_yaml_items(inner):
            key_value = _split_simple_yaml_key_value(item)
            if key_value is None:
                continue
            key, raw_item_value = key_value
            out[str(_coerce_simple_yaml_scalar(key))] = _coerce_simple_yaml_scalar(raw_item_value)
        return out
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_simple_yaml_block(
    lines: list[tuple[int, str]],
    start_index: int,
    block_indent: int,
) -> tuple[Any, int]:
    if start_index >= len(lines):
        return {}, start_index
    current_indent, current_content = lines[start_index]
    if current_indent < block_indent:
        return {}, start_index
    if current_content == "-" or current_content.startswith("- "):
        return _parse_simple_yaml_list(lines, start_index, current_indent)
    return _parse_simple_yaml_mapping_block(lines, start_index, current_indent)


def _parse_simple_yaml_mapping_block(
    lines: list[tuple[int, str]],
    start_index: int,
    block_indent: int,
) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    index = int(start_index)
    while index < len(lines):
        indent, content = lines[index]
        if indent < block_indent:
            break
        if indent > block_indent:
            break
        if content == "-" or content.startswith("- "):
            break
        key_value = _split_simple_yaml_key_value(content)
        if key_value is None:
            index += 1
            continue
        key, raw_value = key_value
        index += 1
        if raw_value == "":
            if index < len(lines) and lines[index][0] > indent:
                value, index = _parse_simple_yaml_block(lines, index, lines[index][0])
            else:
                value = {}
        else:
            value = _coerce_simple_yaml_scalar(raw_value)
        out[str(key)] = value
    return out, index


def _parse_simple_yaml_list(
    lines: list[tuple[int, str]],
    start_index: int,
    block_indent: int,
) -> tuple[list[Any], int]:
    out: list[Any] = []
    index = int(start_index)
    while index < len(lines):
        indent, content = lines[index]
        if indent < block_indent:
            break
        if indent > block_indent:
            break
        if not (content == "-" or content.startswith("- ")):
            break
        item_text = "" if content == "-" else content[2:].strip()
        index += 1
        if item_text == "":
            if index < len(lines) and lines[index][0] > indent:
                item_value, index = _parse_simple_yaml_block(lines, index, lines[index][0])
            else:
                item_value = ""
        else:
            key_value = _split_simple_yaml_key_value(item_text)
            if key_value is not None and not item_text.startswith(("{", "[")):
                key, raw_value = key_value
                item_map: dict[str, Any] = {}
                if raw_value == "":
                    if index < len(lines) and lines[index][0] > indent:
                        child_value, index = _parse_simple_yaml_block(lines, index, lines[index][0])
                    else:
                        child_value = {}
                    item_map[str(key)] = child_value
                else:
                    item_map[str(key)] = _coerce_simple_yaml_scalar(raw_value)
                if index < len(lines) and lines[index][0] > indent:
                    continuation, index = _parse_simple_yaml_block(lines, index, lines[index][0])
                    if isinstance(continuation, dict):
                        item_map.update(continuation)
                item_value = item_map
            else:
                item_value = _coerce_simple_yaml_scalar(item_text)
                if index < len(lines) and lines[index][0] > indent:
                    _, index = _parse_simple_yaml_block(lines, index, lines[index][0])
        out.append(item_value)
    return out, index


def _load_simple_yaml_mapping(file_path: str) -> Dict[str, Any]:
    """Fallback parser for the project config YAML subset."""

    prepared_lines: list[tuple[int, str]] = []
    with Path(file_path).open("r", encoding="utf-8") as file:
        for raw_line in file:
            if raw_line.strip() == "" or raw_line.lstrip().startswith("#"):
                continue
            content_without_comment = _strip_simple_yaml_comment(raw_line)
            if content_without_comment.strip() == "":
                continue
            indent = len(content_without_comment) - len(content_without_comment.lstrip(" "))
            prepared_lines.append((indent, content_without_comment.strip()))

    if not prepared_lines:
        return {}
    parsed, _ = _parse_simple_yaml_block(prepared_lines, 0, prepared_lines[0][0])
    return dict(parsed) if isinstance(parsed, dict) else {}


def _simple_yaml_scalar_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")
    return f'"{text}"'


def _simple_yaml_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * int(indent)
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_simple_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_simple_yaml_scalar_text(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.extend(_simple_yaml_lines(item, indent=indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(_simple_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_simple_yaml_scalar_text(item)}")
        return lines
    return [f"{prefix}{_simple_yaml_scalar_text(value)}"]


def dump_yaml_file(data: Dict[str, Any], file_path: str) -> None:
    """Write project config YAML with PyYAML when available, otherwise a safe subset dumper."""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        yaml = importlib.import_module("yaml")
        if hasattr(yaml, "safe_dump"):
            with path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
            return
    except ModuleNotFoundError:
        pass
    path.write_text("\n".join(_simple_yaml_lines(dict(data))) + "\n", encoding="utf-8")


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Loads one YAML file and returns its parsed dictionary.
    INPUT
    file_path: The path of the YAML file.
    OUTPUT
    A dictionary containing the parsed YAML content.
    EXAMPLE OUTPUT
    {"total_slots": 30, "slot_duration_ms": 1.0}
    """
    path = Path(file_path)
    if not path.exists():
        return {}
    try:
        yaml = importlib.import_module("yaml")
    except ModuleNotFoundError:
        return _load_simple_yaml_mapping(str(path))

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    return data


def load_all_configs(config_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads all framework configuration files from one config directory.
    INPUT
    config_dir: The directory containing simulator YAML configuration files.
    OUTPUT
    A dictionary containing sim, cell, ue, ric, and beamforming sections.
    EXAMPLE OUTPUT
    {
        "sim": {...},
        "cell": {...},
        "ue": {...},
        "ric": {...},
        "beamforming": {...}
    }
    """
    config_path = Path(config_dir)

    return {
        "sim": load_yaml_file(str(config_path / "sim.yaml")),
        "cell": load_yaml_file(str(config_path / "cell.yaml")),
        "ue": load_yaml_file(str(config_path / "ue.yaml")),
        "ric": load_yaml_file(str(config_path / "ric.yaml")),
        "beamforming": load_yaml_file(str(config_path / "beamforming.yaml")),
        "scenario": load_yaml_file(str(config_path / "scenario.yaml")),
    }
