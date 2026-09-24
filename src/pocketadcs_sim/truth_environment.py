"""Integrated, library-backed orbital and environmental truth model."""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
import numpy.typing as npt
import ppigrf
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time
from astropy.utils import iers
from Basilisk.simulation import GravityGradientEffector, eclipse, spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simHelpers, simIncludeGravBody
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path
from scipy.spatial.transform import Rotation

from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TruthEnvironmentConfig:
    """Epoch and model fidelity for an Earth-orbit truth run."""

    epoch_utc: str
    gravity_degree: int = 20

    def __post_init__(self) -> None:
        Time(self.epoch_utc, scale="utc")
        if not 2 <= self.gravity_degree <= 70:
            raise ValueError("gravity_degree must be between 2 and 70")


@dataclass(frozen=True, slots=True)
class TruthEnvironmentHistory:
    """Time-correlated truth values with explicit GCRS/ITRS and SI conventions."""

    time_s: FloatArray
    position_gcrs_m: FloatArray
    velocity_gcrs_m_s: FloatArray
    position_itrs_m: FloatArray
    geodetic_latitude_deg: FloatArray
    geodetic_longitude_deg: FloatArray
    geodetic_height_m: FloatArray
    mrp_body_from_gcrs: FloatArray
    angular_velocity_body_rad_s: FloatArray
    sun_position_gcrs_m: FloatArray
    sun_direction_gcrs: FloatArray
    sun_direction_body: FloatArray
    illumination_fraction: FloatArray
    magnetic_field_itrs_t: FloatArray
    magnetic_field_gcrs_t: FloatArray
    magnetic_field_body_t: FloatArray
    gravity_gradient_torque_body_nm: FloatArray


def _igrf14_itrs(position_itrs_m: FloatArray, times: Time) -> FloatArray:
    fields = np.empty_like(position_itrs_m)
    datetimes = times.to_datetime(timezone=None)
    for index, (position, epoch) in enumerate(zip(position_itrs_m, datetimes, strict=True)):
        x, y, z = position / 1000.0
        radius = float(np.linalg.norm([x, y, z]))
        colatitude = float(np.degrees(np.arccos(z / radius)))
        longitude = float(np.degrees(np.arctan2(y, x)))
        radial, south, east = ppigrf.igrf_gc(radius, colatitude, longitude, epoch)
        br = float(np.asarray(radial).item())
        bt = float(np.asarray(south).item())
        bp = float(np.asarray(east).item())
        theta = np.radians(colatitude)
        phi = np.radians(longitude)
        radial_axis = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        south_axis = np.array(
            [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
        )
        east_axis = np.array([-np.sin(phi), np.cos(phi), 0.0])
        fields[index] = (br * radial_axis + bt * south_axis + bp * east_axis) * 1.0e-9
    return fields


def _itrs_vectors_to_gcrs(vectors_itrs: FloatArray, times: Time) -> FloatArray:
    itrs = ITRS(
        CartesianRepresentation(vectors_itrs.T * u.T),
        obstime=times,
    )
    return np.asarray(itrs.transform_to(GCRS(obstime=times)).cartesian.xyz.to_value(u.T)).T


def run_truth_environment(
    vehicle: SpacecraftConfig,
    initial: InitialState,
    timing: SimulationConfig,
    environment: TruthEnvironmentConfig,
) -> TruthEnvironmentHistory:
    """Run Earth 6-DOF truth with SPICE, GGM03S gravity, eclipse, and IGRF-14."""

    task_name = "truthEnvironmentTask"
    simulation = SimulationBaseClass.SimBaseClass()
    process = simulation.CreateNewProcess("truthEnvironmentProcess")
    process.addTask(simulation.CreateNewTask(task_name, macros.sec2nano(timing.step_s)))

    vehicle_model = spacecraft.Spacecraft()
    vehicle_model.ModelTag = "pocketQube"
    vehicle_model.hub.mHub = vehicle.mass_kg
    vehicle_model.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    vehicle_model.hub.IHubPntBc_B = simHelpers.np2EigenMatrix3d(
        vehicle.inertia_kg_m2.ravel().tolist()
    )
    vehicle_model.hub.r_CN_NInit = initial.position_m.tolist()
    vehicle_model.hub.v_CN_NInit = initial.velocity_m_s.tolist()
    vehicle_model.hub.sigma_BNInit = initial.mrp_body_from_inertial.reshape(3, 1).tolist()
    vehicle_model.hub.omega_BN_BInit = (
        initial.angular_velocity_body_rad_s.reshape(3, 1).tolist()
    )

    gravity = simIncludeGravBody.gravBodyFactory()
    bodies = gravity.createBodies(["earth", "sun"])
    earth = bodies["earth"]
    earth.isCentralBody = True
    earth.useSphericalHarmonicsGravityModel(
        str(get_path(DataFile.LocalGravData.GGM03S)), environment.gravity_degree
    )
    gravity.addBodiesTo(vehicle_model)
    spice = gravity.createSpiceInterface(
        time=Time(environment.epoch_utc, scale="utc").strftime(
            "%Y %b %d %H:%M:%S.%f (UTC)"
        ),
        epochInMsg=True,
    )
    spice.zeroBase = "earth"
    earth_message = spice.planetStateOutMsgs[0]
    sun_message = spice.planetStateOutMsgs[1]

    shadow = eclipse.Eclipse()
    shadow.ModelTag = "earthEclipse"
    shadow.sunInMsg.subscribeTo(sun_message)
    shadow.addPlanetToModel(earth_message)
    shadow.addSpacecraftToModel(vehicle_model.scStateOutMsg)

    gravity_gradient = GravityGradientEffector.GravityGradientEffector()
    gravity_gradient.ModelTag = vehicle_model.ModelTag
    gravity_gradient.addPlanetName(earth.planetName)
    vehicle_model.addDynamicEffector(gravity_gradient)

    state_recorder = vehicle_model.scStateOutMsg.recorder()
    sun_recorder = sun_message.recorder()
    eclipse_recorder = shadow.eclipseOutMsgs[0].recorder()
    gravity_gradient_recorder = gravity_gradient.gravityGradientOutMsg.recorder()

    for model in (
        spice,
        shadow,
        gravity_gradient,
        vehicle_model,
        state_recorder,
        sun_recorder,
        eclipse_recorder,
        gravity_gradient_recorder,
    ):
        simulation.AddModelToTask(task_name, model)

    simulation.InitializeSimulation()
    simulation.ConfigureStopTime(macros.sec2nano(timing.duration_s))
    simulation.ExecuteSimulation()

    time_s = np.asarray(state_recorder.times(), dtype=float) * macros.NANO2SEC
    iers.conf.auto_download = False
    times = Time(environment.epoch_utc, scale="utc") + time_s * u.s
    position_gcrs = np.asarray(state_recorder.r_BN_N, dtype=float)
    gcrs = GCRS(
        CartesianRepresentation(position_gcrs.T * u.m),
        obstime=times,
    )
    position_itrs = np.asarray(
        gcrs.transform_to(ITRS(obstime=times)).cartesian.xyz.to_value(u.m)
    ).T
    locations = EarthLocation.from_geocentric(
        position_itrs[:, 0] * u.m,
        position_itrs[:, 1] * u.m,
        position_itrs[:, 2] * u.m,
    )
    longitude, latitude, height = locations.to_geodetic(ellipsoid="WGS84")
    magnetic_itrs = _igrf14_itrs(position_itrs, times)
    magnetic_gcrs = _itrs_vectors_to_gcrs(magnetic_itrs, times)
    mrp_body_from_gcrs = np.asarray(state_recorder.sigma_BN, dtype=float)
    body_from_gcrs = Rotation.from_mrp(mrp_body_from_gcrs).inv()
    sun_position_gcrs = np.asarray(sun_recorder.PositionVector, dtype=float)
    sun_displacement = sun_position_gcrs - position_gcrs
    sun_direction_gcrs = sun_displacement / np.linalg.norm(
        sun_displacement, axis=1, keepdims=True
    )

    return TruthEnvironmentHistory(
        time_s=time_s,
        position_gcrs_m=position_gcrs,
        velocity_gcrs_m_s=np.asarray(state_recorder.v_BN_N, dtype=float),
        position_itrs_m=position_itrs,
        geodetic_latitude_deg=np.asarray(latitude.to_value(u.deg), dtype=float),
        geodetic_longitude_deg=np.asarray(longitude.to_value(u.deg), dtype=float),
        geodetic_height_m=np.asarray(height.to_value(u.m), dtype=float),
        mrp_body_from_gcrs=mrp_body_from_gcrs,
        angular_velocity_body_rad_s=np.asarray(state_recorder.omega_BN_B, dtype=float),
        sun_position_gcrs_m=sun_position_gcrs,
        sun_direction_gcrs=sun_direction_gcrs,
        sun_direction_body=body_from_gcrs.apply(sun_direction_gcrs),
        illumination_fraction=np.asarray(eclipse_recorder.shadowFactor, dtype=float),
        magnetic_field_itrs_t=magnetic_itrs,
        magnetic_field_gcrs_t=magnetic_gcrs,
        magnetic_field_body_t=body_from_gcrs.apply(magnetic_gcrs),
        gravity_gradient_torque_body_nm=np.asarray(
            gravity_gradient_recorder.gravityGradientTorque_B, dtype=float
        ),
    )
