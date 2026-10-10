"""Rotor configuration for the TGR 29-12 based three-wheel set.

All numbers come from the **Motor Parameter Table** on page 1 of
``TGR-29-12-Datasheet-Rev-B.pdf`` (thingap.com, Rev B). That table was read from
a rendered image of the page, not from ``pdftotext``: the text layer emits the
units column and the values column as two separate blocks, so a text extraction
silently misaligns every label from its number. Both the atmosphere and vacuum
columns are recorded here because they differ by ~5x in continuous torque and
choosing the wrong one is a silent 5x error in control authority.

Nothing in this module is a *derived* limit. In particular
:data:`TGR2912.continuous_torque_vacuum_nm` is a torque capability, not a
verified stall-free acceleration: the datasheet states no stall torque and no
speed-vs-torque derating curve, so :attr:`TGR2912.acceleration_cap_rad_s2` is
an explicit engineering parameter chosen by the user (see the design doc), not
a datasheet claim.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DATASHEET_SOURCE",
    "RotorConfig",
    "TGR2912",
    "WheelSet",
    "default_wheel_set",
]

DATASHEET_SOURCE = (
    "TGR 29-12 Reaction Wheel, Micro Motor, Data Sheet Rev B, "
    "Motor Parameter Table, p.1 (https://www.thingap.com/wp-content/uploads/2023/02/"
    "TGR-29-12-Datasheet-Rev-B.pdf)"
)

#: Body axes of the three wheels, one per orthogonal axis.
_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class RotorConfig:
    """One rotor's datasheet parameters and the limits chosen for the sim.

    Attributes
    ----------
    rotor_inertia_kg_m2:
        ``Rotor Inertia`` from the table, in kg m^2. This is the *only* inertia
        the model attributes to a wheel: no flywheel or extra inertia is added.
    max_speed_rpm:
        ``Max Continuous Speed``, both columns, in rpm.
    continuous_torque_nm:
        Continuous torque at max speed. Kept as a pair because the atmosphere
        and vacuum columns differ substantially.
    peak_torque_1s_nm / peak_torque_3s_nm:
        ``Peak Torque`` rows, which are short-duration ratings.
    acceleration_cap_rad_s2:
        Engineering parameter. Not from the datasheet; see the module docstring.
    """

    name: str
    rotor_inertia_kg_m2: float
    rotor_mass_kg: float
    max_speed_rpm: float
    continuous_torque_atm_nm: float
    continuous_torque_vac_nm: float
    peak_torque_1s_atm_nm: float
    peak_torque_1s_vac_nm: float
    peak_torque_3s_atm_nm: float
    peak_torque_3s_vac_nm: float
    magnetic_poles: int
    torque_constant_nm_per_arms: float
    resistance_ohm: float
    inductance_uh: float
    electrical_frequency_hz_at_max_speed: float
    acceleration_cap_rad_s2: float

    @property
    def max_speed_rad_s(self) -> float:
        """``Max Continuous Speed`` in rad/s."""
        return self.max_speed_rpm * 2.0 * 3.141592653589793 / 60.0

    @property
    def continuous_torque_nm(self) -> float:
        """Vacuum continuous torque, the operating environment for this study."""
        return self.continuous_torque_vac_nm

    @property
    def peak_torque_nm(self) -> float:
        """The 1 s vacuum peak, used only for reporting headroom."""
        return self.peak_torque_1s_vac_nm

    @property
    def max_acceleration_from_continuous_torque_rad_s2(self) -> float:
        """Acceleration implied by the continuous torque -- a capability, not a cap.

        Reported so a chosen :attr:`acceleration_cap_rad_s2` can be compared
        against the hardware's own torque-limited acceleration. The datasheet
        gives no stall torque, so exceeding this is not modelled as stall.
        """
        return self.continuous_torque_nm / self.rotor_inertia_kg_m2

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable record including provenance."""
        return {
            "name": self.name,
            "source": DATASHEET_SOURCE,
            "rotor_inertia_kg_m2": self.rotor_inertia_kg_m2,
            "rotor_mass_kg": self.rotor_mass_kg,
            "max_speed_rpm": self.max_speed_rpm,
            "max_speed_rad_s": self.max_speed_rad_s,
            "continuous_torque_atm_nm": self.continuous_torque_atm_nm,
            "continuous_torque_vac_nm": self.continuous_torque_vac_nm,
            "continuous_torque_nm_used": self.continuous_torque_nm,
            "peak_torque_1s_atm_nm": self.peak_torque_1s_atm_nm,
            "peak_torque_1s_vac_nm": self.peak_torque_1s_vac_nm,
            "peak_torque_3s_atm_nm": self.peak_torque_3s_atm_nm,
            "peak_torque_3s_vac_nm": self.peak_torque_3s_vac_nm,
            "magnetic_poles": self.magnetic_poles,
            "torque_constant_nm_per_arms": self.torque_constant_nm_per_arms,
            "resistance_ohm": self.resistance_ohm,
            "inductance_uh": self.inductance_uh,
            "electrical_frequency_hz_at_max_speed": self.electrical_frequency_hz_at_max_speed,
            "acceleration_cap_rad_s2": self.acceleration_cap_rad_s2,
            "max_acceleration_from_continuous_torque_rad_s2": (
                self.max_acceleration_from_continuous_torque_rad_s2
            ),
            "acceleration_cap_is_datasheet_claim": False,
        }


#: Values read from the page-1 Motor Parameter Table (Rev B).
TGR2912 = RotorConfig(
    name="TGR 29-12",
    rotor_inertia_kg_m2=3.27e-6,
    rotor_mass_kg=0.028,
    max_speed_rpm=16000.0,
    continuous_torque_atm_nm=0.0135,
    continuous_torque_vac_nm=0.0028,
    peak_torque_1s_atm_nm=0.036,
    peak_torque_1s_vac_nm=0.034,
    peak_torque_3s_atm_nm=0.024,
    peak_torque_3s_vac_nm=0.019,
    magnetic_poles=6,
    torque_constant_nm_per_arms=0.0056,
    resistance_ohm=1.019,
    inductance_uh=7.13,
    electrical_frequency_hz_at_max_speed=300.0,
    # Engineering parameter, NOT a datasheet value. 200 rad/s^2 is ~23% of the
    # acceleration the continuous torque could produce (~857 rad/s^2), chosen as
    # a conservative command-slew limit. No stall model is implemented, so this
    # is a commanded bound and is reported as such in the results.
    acceleration_cap_rad_s2=200.0,
)


@dataclass(frozen=True)
class WheelSet:
    """Three orthogonal, body-axis-aligned wheels under perfect commutation.

    Perfect commutation is the user's explicit simplifying assumption, so the
    wheel set is a plain diagonal allocation: wheel ``i`` produces torque about
    body axis ``i`` and nothing else. No null-space optimization is modeled,
    because a real reaction-wheel triad cannot null a single vector anyway.
    """

    rotors: tuple[RotorConfig, ...]
    axes: tuple[tuple[float, float, float], ...]
    name: str = "TGR 29-12 triad"

    def __post_init__(self) -> None:
        if len(self.rotors) != 3 or len(self.axes) != 3:
            raise ValueError("a wheel set needs exactly three rotors and three axes")

    @property
    def inertia(self) -> float:
        """Common rotor inertia; the three rotors are identical parts."""
        values = {rotor.rotor_inertia_kg_m2 for rotor in self.rotors}
        if len(values) != 1:
            raise ValueError("mixed rotor inertias are not supported by this model")
        return values.pop()

    @property
    def acceleration_cap_rad_s2(self) -> float:
        """The commanded acceleration cap, shared by identical rotors."""
        values = {rotor.acceleration_cap_rad_s2 for rotor in self.rotors}
        if len(values) != 1:
            raise ValueError("mixed acceleration caps are not supported by this model")
        return values.pop()

    @property
    def max_speed_rad_s(self) -> float:
        """The speed bound, shared by identical rotors."""
        values = {rotor.max_speed_rad_s for rotor in self.rotors}
        if len(values) != 1:
            raise ValueError("mixed speed limits are not supported by this model")
        return values.pop()

    @property
    def continuous_torque_nm(self) -> float:
        """Vacuum continuous torque, shared by identical rotors."""
        values = {rotor.continuous_torque_nm for rotor in self.rotors}
        if len(values) != 1:
            raise ValueError("mixed torques are not supported by this model")
        return values.pop()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record of the whole set."""
        return {
            "name": self.name,
            "source": DATASHEET_SOURCE,
            "commutation": "perfect, three orthogonal body-axis-aligned wheels",
            "axes": [list(axis) for axis in self.axes],
            "rotors": [rotor.to_dict() for rotor in self.rotors],
        }


def default_wheel_set(
    acceleration_cap_rad_s2: float | None = None,
) -> WheelSet:
    """Return the default three-wheel TGR 29-12 set on the body principal axes.

    ``acceleration_cap_rad_s2`` overrides the datasheet-annotated default cap;
    pass it to explore the commanded-acceleration bound without editing the
    datasheet record.
    """
    if acceleration_cap_rad_s2 is None:
        rotor = TGR2912
    else:
        if acceleration_cap_rad_s2 <= 0.0:
            raise ValueError("acceleration cap must be positive")
        rotor = RotorConfig(
            **{
                **{f: getattr(TGR2912, f) for f in TGR2912.__dataclass_fields__},
                "acceleration_cap_rad_s2": float(acceleration_cap_rad_s2),
            }
        )
    return WheelSet(rotors=(rotor, rotor, rotor), axes=_AXES)
