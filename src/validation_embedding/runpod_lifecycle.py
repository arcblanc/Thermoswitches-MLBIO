"""RunPod pod stop helpers — call after any standalone job on a pod."""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request

from validation_embedding.config import load_llm_settings

# Orchestrator sets this so child GenerRNA/BiRNA processes do not stop the pod
# before the full pipeline finishes.
SKIP_TERMINATE_ENV = "RUNPOD_SKIP_TERMINATE"

# SSH usernames look like "{podId}-{hexSuffix}"; REST API wants podId only.
_SSH_USER_POD_ID = re.compile(r"^([a-zA-Z0-9]+)-[a-fA-F0-9]{6,}$")


def normalize_pod_id(pod_id: str | None) -> str | None:
    """Strip SSH-username suffixes so the REST API receives a pod id."""
    if not pod_id:
        return None
    pod_id = pod_id.strip()
    match = _SSH_USER_POD_ID.match(pod_id)
    if match:
        return match.group(1)
    return pod_id


def should_skip_terminate() -> bool:
    """Return whether orchestrator children should skip pod termination."""
    return os.environ.get(SKIP_TERMINATE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def runpod_terminate_if_configured(*, force: bool = False) -> bool:
    """Stop the RunPod pod when STORAGE_TARGET=runpod and auto-terminate is on.

    Returns True if a stop request was sent successfully.
    When RUNPOD_SKIP_TERMINATE is set (orchestrator children), this is a no-op
    unless force=True.
    """
    if not force and should_skip_terminate():
        print("RUNPOD_SKIP_TERMINATE set — child job will not stop the pod")
        return False

    settings = load_llm_settings()
    if settings.storage_target != "runpod":
        return False
    if not settings.runpod_auto_terminate:
        print("RUNPOD_AUTO_TERMINATE=false — leaving pod running")
        return False

    pod_id = normalize_pod_id(settings.runpod_pod_id)
    if not settings.runpod_api_key or not pod_id:
        print("RUNPOD_API_KEY or RUNPOD_POD_ID missing; skipping pod terminate")
        return False

    if pod_id != (settings.runpod_pod_id or "").strip():
        print(
            f"Normalized RUNPOD_POD_ID {settings.runpod_pod_id!r} -> {pod_id!r} "
            "(API id, not SSH username)"
        )

    url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
    request = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.runpod_api_key}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode()
        print(f"RunPod stop requested for {pod_id}: {body}")
        return True
    except urllib.error.HTTPError as exc:
        print(f"RunPod stop failed ({exc.code}): {exc.read().decode()}")
        return False
    except Exception as exc:
        print(f"RunPod stop failed: {exc}")
        return False
