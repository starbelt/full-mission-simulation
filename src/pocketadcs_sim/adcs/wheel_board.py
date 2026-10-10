"""Stand-in for the three-wheel control board.

The board is the actuator interface of the prototype: it accepts a commanded
body-frame torque, enforces what the hardware can actually do, applies the
result to its rotors, and reports back what it really did. It is deliberately a
*stand-in*, not a hardware model — it executes commands, keeps continuous rotor
state, and reports commanded versus applied torque honestly, but it contains no
driver, no bus and no failure modes beyond the documented limits.

Two limits are enforced, in this order:

1. **Commanded acceleration cap** — ``|u_i / J| <= acceleration_cap_rad_s2``.
   This is an *engineering parameter*, not a datasheet quantity and not a
   stall model (see :mod:`pocketadcs_sim.adcs.rotors`); the datasheet gives no
   stall torque, so nothing here claims the cap prevents stall.
2. **Speed bound** — a wheel at ``max_speed_rad_s`` refuses torque that would
   push it further out, but may still be braked. Clipping at the bound in both
   directions would freeze a wheel that is overspeeding and could not recover.

A ``None`` command models a command timeout: torque goes to zero and rotor speed
is *preserved*. Zero-filling wheel state on timeout is the classic way to turn a
10-second comms dropout into a full attitude excursion, because the stored
momentum silently disappears instead of being re-injected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pocketadcs_sim.adcs.rotors import DATASHEET_SOURCE, WheelSet, default_wheel_set

__all__ = ["CommandTimeout", "WheelBoard"]

FloatArray = npt.NDArray[np.float64]


class CommandTimeout(Exception):
    """Raised by callers that choose to treat a missing command as fatal.

    The board itself handles a timeout by coasting, so this exists for the
    surrounding loop to distinguish "no command arrived" from "a zero torque
    was commanded" — two states that would otherwise be indistinguishable.
    """


@dataclass(frozen=True)
class _Limits:
    """Per-wheel limits, all shared because the rotors are identical parts."""

    inertia: float
    acceleration_cap: float
    max_speed: float


class WheelBoard:
    """Three-wheel board with continuous rotor state and honest reporting."""

    def __init__(
        self,
        wheel_set: WheelSet | None = None,
        acceleration_cap_rad_s2: float | None = None,
    ) -> None:
        self.wheel_set = wheel_set if wheel_set is not None else default_wheel_set(
            acceleration_cap_rad_s2
        )
        self._axes = np.asarray(self.wheel_set.axes, dtype=float)
        self._limits = _Limits(
            inertia=self.wheel_set.inertia,
            acceleration_cap=self.wheel_set.acceleration_cap_rad_s2,
            max_speed=self.wheel_set.max_speed_rad_s,
        )
        self.wheel_speed_rad_s = np.zeros(3)
        self.commanded_torque_nm = np.zeros(3)
        self.applied_torque_nm = np.zeros(3)
        self.saturated_per_wheel = np.zeros(3, dtype=bool)
        self.timed_out = False
        self.time_s = 0.0

    @property
    def max_speed_rad_s(self) -> float:
        """The commanded rotor speed bound."""
        return self._limits.max_speed

    @property
    def saturated(self) -> bool:
        """True when any wheel was limited on the most recent update."""
        return bool(self.saturated_per_wheel.any())

    def set_wheel_speed(self, wheel_speed_rad_s: npt.ArrayLike) -> None:
        """Set rotor speed directly (initialization, or a state injection in HIL)."""
        speed = np.asarray(wheel_speed_rad_s, dtype=float).reshape(3)
        if not np.all(np.isfinite(speed)):
            raise ValueError("wheel speed must be finite")
        self.wheel_speed_rad_s = speed.copy()

    def update(
        self,
        torque_nm: npt.ArrayLike | None,
        dt_s: float,
    ) -> FloatArray:
        """Apply one board cycle and return the torque actually delivered.

        ``torque_nm`` is a body-frame torque command; ``None`` models a command
        timeout, which zeroes the torque but preserves rotor speed. Returns the
        applied per-wheel torque, so the plant can be driven with what the board
        was able to do rather than with what it was asked to do.
        """
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be a positive finite number")

        self.timed_out = torque_nm is None
        if torque_nm is None:
            self.commanded_torque_nm = np.zeros(3)
        else:
            commanded = np.asarray(torque_nm, dtype=float).reshape(-1)
            if commanded.shape != (3,):
                raise ValueError("torque_nm must have shape (3,)")
            if not np.all(np.isfinite(commanded)):
                raise ValueError("torque_nm must be finite")
            self.commanded_torque_nm = commanded.copy()

        applied = self._apply_limits(self.commanded_torque_nm, dt)
        self.applied_torque_nm = applied
        # Omega is relative to the body: a pure command integral, u/J.
        self.wheel_speed_rad_s = self.wheel_speed_rad_s + applied / self._limits.inertia * dt
        self.time_s += dt
        return applied

    def _apply_limits(self, commanded: FloatArray, dt: float) -> FloatArray:
        """Return the torque after the acceleration cap and the speed bound."""
        inertia = self._limits.inertia

        # 1. Acceleration cap: bounds the torque this step may apply.
        cap_torque = self._limits.acceleration_cap * inertia
        applied = np.clip(commanded, -cap_torque, cap_torque)

        # 2. Speed bound: only blocks torque that would push further out.
        projected = self.wheel_speed_rad_s + applied / inertia * dt
        overspeed = np.abs(projected) > self._limits.max_speed
        same_sign = applied * self.wheel_speed_rad_s > 0.0
        blocked = overspeed & same_sign
        applied = np.where(blocked, 0.0, applied)

        # A wheel is reported saturated when the limit changed what it delivered.
        self.saturated_per_wheel = blocked | (np.abs(applied - commanded) > 0.0)
        return applied

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record of the board configuration."""
        record: dict[str, object] = {
            "source": DATASHEET_SOURCE,
            "commutation": "perfect",
            "axes": [list(axis) for axis in self._axes],
            "rotor_inertia_kg_m2": self._limits.inertia,
            "max_speed_rad_s": self._limits.max_speed,
            "acceleration_cap_rad_s2": self._limits.acceleration_cap,
            "acceleration_cap_is_datasheet_claim": False,
            "command_timeout_behaviour": "zero torque, rotor speed preserved (coast)",
        }
        return record
