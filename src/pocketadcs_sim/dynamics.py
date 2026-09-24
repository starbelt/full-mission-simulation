"""Basilisk-backed spacecraft truth dynamics.

This module is an adapter around Basilisk. It does not implement orbital or
rigid-body equations locally. Public inputs and outputs use SI units.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simHelpers, simIncludeGravBody

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SpacecraftConfig:
    """Rigid PocketQube hub properties in SI units."""

    mass_kg: float
    inertia_kg_m2: FloatArray

    def __post_init__(self) -> None:
        inertia = np.asarray(self.inertia_kg_m2, dtype=float)
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if inertia.shape != (3, 3):
            raise ValueError("inertia_kg_m2 must have shape (3, 3)")
        if not np.allclose(inertia, inertia.T):
            raise ValueError("inertia_kg_m2 must be symmetric")
        if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
            raise ValueError("inertia_kg_m2 must be positive definite")
        object.__setattr__(self, "inertia_kg_m2", inertia)


@dataclass(frozen=True, slots=True)
class InitialState:
    """Initial Earth-centered inertial translational and rotational state."""

    position_m: FloatArray
    velocity_m_s: FloatArray
    mrp_body_from_inertial: FloatArray
    angular_velocity_body_rad_s: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "position_m",
            "velocity_m_s",
            "mrp_body_from_inertial",
            "angular_velocity_body_rad_s",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Fixed-step truth-simulation timing in seconds."""

    duration_s: float
    step_s: float

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.step_s <= 0.0:
            raise ValueError("step_s must be positive")
        steps = self.duration_s / self.step_s
        if not np.isclose(steps, round(steps)):
            raise ValueError("duration_s must be an integer multiple of step_s")


@dataclass(frozen=True, slots=True)
class TruthHistory:
    """Recorded Basilisk spacecraft state history in SI units."""

    time_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    mrp_body_from_inertial: FloatArray
    angular_velocity_body_rad_s: FloatArray


def run_truth_simulation(
    vehicle: SpacecraftConfig,
    initial: InitialState,
    timing: SimulationConfig,
) -> TruthHistory:
    """Run point-mass Earth, six-degree-of-freedom Basilisk dynamics."""

    task_name = "truthDynamicsTask"
    process_name = "truthDynamicsProcess"

    simulation = SimulationBaseClass.SimBaseClass()
    process = simulation.CreateNewProcess(process_name)
    process.addTask(simulation.CreateNewTask(task_name, macros.sec2nano(timing.step_s)))

    spacecraft_model = spacecraft.Spacecraft()
    spacecraft_model.ModelTag = "pocketQube"
    spacecraft_model.hub.mHub = vehicle.mass_kg
    spacecraft_model.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    spacecraft_model.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
        vehicle.inertia_kg_m2.ravel().tolist()
    )
    spacecraft_model.hub.r_CN_NInit = initial.position_m.tolist()
    spacecraft_model.hub.v_CN_NInit = initial.velocity_m_s.tolist()
    spacecraft_model.hub.sigma_BNInit = initial.mrp_body_from_inertial.reshape(3, 1).tolist()
    spacecraft_model.hub.omega_BN_BInit = (
        initial.angular_velocity_body_rad_s.reshape(3, 1).tolist()
    )
    simulation.AddModelToTask(task_name, spacecraft_model)

    gravity = simIncludeGravBody.gravBodyFactory()
    earth = gravity.createEarth()
    earth.isCentralBody = True
    gravity.addBodiesTo(spacecraft_model)

    recorder = spacecraft_model.scStateOutMsg.recorder()
    simulation.AddModelToTask(task_name, recorder)

    simulation.InitializeSimulation()
    simulation.ConfigureStopTime(macros.sec2nano(timing.duration_s))
    simulation.ExecuteSimulation()

    return TruthHistory(
        time_s=np.asarray(recorder.times(), dtype=float) * macros.NANO2SEC,
        position_m=np.asarray(recorder.r_BN_N, dtype=float),
        velocity_m_s=np.asarray(recorder.v_BN_N, dtype=float),
        mrp_body_from_inertial=np.asarray(recorder.sigma_BN, dtype=float),
        angular_velocity_body_rad_s=np.asarray(recorder.omega_BN_B, dtype=float),
    )
