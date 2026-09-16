"""Coordinate transforms used by the paper-style observation/state spaces.

The paper states that every state containing position and heading angle is
represented in both geo coordinates and the observing RSUAV's body frame.

This project keeps the scene convention used by the validated visualizer:
``yaw == 0`` points North.  In the body frame, ``x`` points forward and ``y``
points to the observer's right.  Therefore a point straight ahead has
``body.x > 0`` and ``body.y == 0``.
"""
from __future__ import annotations

import math
from typing import Any, Dict


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def geo_pose(x: float, y: float, yaw: float) -> Dict[str, float]:
    """Return the absolute geo-coordinate representation of a pose."""
    return {"x": float(x), "y": float(y), "yaw": float(yaw)}


def body_pose(observer: Any, x: float, y: float, yaw: float) -> Dict[str, float]:
    """Represent an object's pose in ``observer``'s body coordinate frame.

    With the environment's North-zero heading convention:

    - body ``x``: forward component
    - body ``y``: right-hand lateral component
    - body ``yaw``: object's heading relative to the observer

    ``range`` and ``bearing`` are included as exact derived quantities because
    the paper also uses the polar body-coordinate quantities rho/theta in its
    strike model (Eq. 4).  They are metadata for now; the later FOFE tensor
    encoder can explicitly select which fields become neural-network inputs.
    """
    dx = float(x) - float(observer.x)
    dy = float(y) - float(observer.y)
    psi = float(observer.yaw)

    # Forward/right axes for yaw=0 -> North.
    forward = dx * math.sin(psi) + dy * math.cos(psi)
    right = dx * math.cos(psi) - dy * math.sin(psi)

    return {
        "x": float(forward),
        "y": float(right),
        "yaw": float(wrap_angle(float(yaw) - psi)),
        "range": float(math.hypot(dx, dy)),
        "bearing": float(math.atan2(right, forward)),
    }


def dual_pose(observer: Any, obj: Any, yaw: float | None = None) -> Dict[str, Dict[str, float]]:
    """Return both geo- and body-coordinate pose representations."""
    obj_yaw = float(getattr(obj, "yaw", 0.0) if yaw is None else yaw)
    return {
        "geo": geo_pose(obj.x, obj.y, obj_yaw),
        "body": body_pose(observer, obj.x, obj.y, obj_yaw),
    }
