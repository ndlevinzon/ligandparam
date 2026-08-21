"""Slim IPC payloads for nested twist / scan process pools."""

from __future__ import annotations

from typing import Optional


def slim_scan_result(scan_result: Optional[dict]) -> Optional[dict]:
    """Drop heavy objects from one bond-scan result dict."""
    if not isinstance(scan_result, dict):
        return scan_result
    return {k: v for k, v in scan_result.items() if k != "wf_run"}


def slim_twist_result(twist_result: Optional[dict]) -> Optional[dict]:
    """Drop heavy ``wf_run`` objects so fragment-pool IPC stays picklable."""
    if twist_result is None:
        return None
    slim = {k: v for k, v in twist_result.items() if k != "scans"}
    scans_out = []
    for item in twist_result.get("scans", []) or []:
        if isinstance(item, tuple) and len(item) == 3:
            prefix, idxs, payload = item
            if isinstance(payload, dict):
                payload = {k: v for k, v in payload.items() if k != "wf_run"}
            scans_out.append((prefix, idxs, payload))
        else:
            scans_out.append(item)
    slim["scans"] = scans_out
    return slim
