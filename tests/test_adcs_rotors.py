"""Tests pinning the TGR 29-12 rotor configuration to its datasheet source.

The point of these tests is not arithmetic: it is that a later edit cannot
quietly change a physical constant, and cannot present an engineering parameter
as if the datasheet stated it.
"""

from __future__ import annotations

import math

import pytest

from pocketadcs_sim.adcs import rotors


def test_datasheet_values_match_the_page_one_table() -> None:
    """The Motor Parameter Table values, as read from the rendered page 1."""
    rotor = rotors.TGR2912

    assert rotor.rotor_inertia_kg_m2 == 3.27e-6
    assert rotor.rotor_mass_kg == 0.028
    assert rotor.max_speed_rpm == 16000.0
    assert rotor.continuous_torque_vac_nm == 0.0028
    assert rotor.continuous_torque_atm_nm == 0.0135
    assert rotor.magnetic_poles == 6
    assert rotor.electrical_frequency_hz_at_max_speed == 300.0


def test_vacuum_torque_is_used_not_the_atmosphere_column() -> None:
    """The atmosphere column is ~5x larger; picking it would be a silent 5x error."""
    rotor = rotors.TGR2912

    assert rotor.continuous_torque_nm == rotor.continuous_torque_vac_nm
    assert rotor.continuous_torque_nm != rotor.continuous_torque_atm_nm
    assert rotor.continuous_torque_atm_nm == pytest.approx(
        4.8 * rotor.continuous_torque_vac_nm, rel=0.05
    )


def test_max_speed_converts_to_rad_s() -> None:
    """16000 rpm is 1675.5 rad/s."""
    assert rotors.TGR2912.max_speed_rad_s == pytest.approx(1675.516, abs=1e-3)


def test_acceleration_cap_is_flagged_as_not_a_datasheet_claim() -> None:
    """The cap is an engineering parameter and must never be reported as proven."""
    record = rotors.TGR2912.to_dict()

    assert record["acceleration_cap_is_datasheet_claim"] is False
    assert record["acceleration_cap_rad_s2"] == rotors.TGR2912.acceleration_cap_rad_s2
    # The torque-implied acceleration is a *capability*, reported alongside so a
    # reader can see how much of it the commanded cap actually uses.
    capability = rotors.TGR2912.continuous_torque_nm / rotors.TGR2912.rotor_inertia_kg_m2
    assert record["max_acceleration_from_continuous_torque_rad_s2"] == pytest.approx(capability)
    assert capability == pytest.approx(856.3, rel=1e-3)
    assert rotors.TGR2912.acceleration_cap_rad_s2 < capability


def test_record_carries_source_and_both_environment_columns() -> None:
    """Provenance and the unused atmosphere column both survive into the record."""
    record = rotors.TGR2912.to_dict()

    assert "TGR 29-12" in str(record["source"])
    assert record["continuous_torque_atm_nm"] == 0.0135
    assert record["continuous_torque_vac_nm"] == 0.0028
    assert record["peak_torque_1s_vac_nm"] == 0.034
    assert record["peak_torque_3s_vac_nm"] == 0.019


def test_default_wheel_set_is_three_orthogonal_body_axes() -> None:
    """Three wheels, one per body axis, forming an orthonormal triad."""
    wheel_set = rotors.default_wheel_set()

    assert len(wheel_set.rotors) == 3
    assert wheel_set.axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for axis in wheel_set.axes:
        assert math.isclose(math.sqrt(sum(component**2 for component in axis)), 1.0)
    for first in range(3):
        for second in range(first + 1, 3):
            first_axis = wheel_set.axes[first]
            second_axis = wheel_set.axes[second]
            dot = sum(a * b for a, b in zip(first_axis, second_axis, strict=True))
            assert dot == 0.0


def test_wheel_set_reports_shared_inertia_speed_and_cap() -> None:
    """Identical rotors give a single shared inertia, speed limit and cap."""
    wheel_set = rotors.default_wheel_set()

    assert wheel_set.inertia == 3.27e-6
    assert wheel_set.max_speed_rad_s == pytest.approx(1675.516, abs=1e-3)
    assert wheel_set.acceleration_cap_rad_s2 == 200.0
    assert wheel_set.continuous_torque_nm == 0.0028


def test_acceleration_cap_can_be_overridden_without_touching_the_datasheet() -> None:
    """An override changes the set's cap and leaves the module-level record alone."""
    wheel_set = rotors.default_wheel_set(acceleration_cap_rad_s2=50.0)

    assert wheel_set.acceleration_cap_rad_s2 == 50.0
    assert rotors.TGR2912.acceleration_cap_rad_s2 == 200.0
    assert wheel_set.rotors[0].rotor_inertia_kg_m2 == 3.27e-6

    with pytest.raises(ValueError):
        rotors.default_wheel_set(acceleration_cap_rad_s2=0.0)


def test_wheel_set_record_is_json_serializable_and_complete() -> None:
    """The set record documents commutation and every rotor, ready for JSON output."""
    import json

    record = json.loads(json.dumps(rotors.default_wheel_set().to_dict()))

    assert record["commutation"].startswith("perfect")
    assert len(record["rotors"]) == 3
    assert len(record["axes"]) == 3
    assert record["rotors"][0]["rotor_inertia_kg_m2"] == 3.27e-6


def test_wheel_set_rejects_mixed_rotors() -> None:
    """Silent averaging of mismatched wheels would hide a configuration error."""
    fast = rotors.default_wheel_set().rotors[0]
    slow = rotors.RotorConfig(
        **{**{f: getattr(fast, f) for f in fast.__dataclass_fields__},
           "rotor_inertia_kg_m2": 5.0e-6}
    )
    wheel_set = rotors.WheelSet(rotors=(fast, slow, fast), axes=rotors.default_wheel_set().axes)

    with pytest.raises(ValueError, match="mixed rotor inertias"):
        _ = wheel_set.inertia
