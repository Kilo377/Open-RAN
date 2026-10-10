"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: debug.py
Description: This file defines the global debug configuration used by the
project. All modules should rely on the same debug flag mechanism so that
debug information can be enabled or disabled consistently.
"""

import os


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = str(raw_value).strip().lower()
    if normalized in {"0", "false", "off", "no", "n"}:
        return False
    if normalized in {"1", "true", "on", "yes", "y"}:
        return True
    return bool(default)


class DebugConfig:
    """
    Stores the global debug flag for the project.
    INPUT
    No explicit input.
    OUTPUT
    A class-level debug switch that can be shared by all modules.
    EXAMPLE OUTPUT
    DebugConfig.DEBUG = True
    """

    DEBUG: bool = False
    ORCHESTRATION_SUMMARY: bool = False
    CONCISE_SLOT_METRICS: bool = False
    SLOT_METRICS: bool = _env_flag("ORAN_SLOT_METRICS", True)


def debug_print(message: str) -> None:
    """
    Prints a debug message only when the global debug flag is enabled.
    INPUT
    message: The debug message to be printed.
    OUTPUT
    None.
    EXAMPLE OUTPUT
    [DEBUG] Slot 1 started.
    """
    if DebugConfig.DEBUG:
        print(f"[DEBUG] {message}")


def orchestration_print(message: str) -> None:
    """
    Prints one orchestration-summary message when summary mode is enabled.
    """
    if DebugConfig.ORCHESTRATION_SUMMARY:
        print(message)
