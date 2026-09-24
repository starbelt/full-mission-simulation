"""Create an overview of all currently simulated truth-environment factors."""

from pathlib import Path

import matplotlib
import numpy as np
from Basilisk.utilities import simIncludeGravBody

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig  # noqa: E402
from pocketadcs_sim.truth_environment import (  # noqa: E402
    TruthEnvironmentConfig,
    run_truth_environment,
)

EARTH_RADIUS_M = 6_378_137.0
ALTITUDE_M = 550_000.0
ORBIT_RADIUS_M = EARTH_RADIUS_M + ALTITUDE_M


def local_magnetic_components(
    field_itrs_t: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    east = np.column_stack((-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)))
    north = np.column_stack(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        )
    )
    up = np.column_stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )
    return (
        np.sum(field_itrs_t * north, axis=1) * 1.0e6,
        np.sum(field_itrs_t * east, axis=1) * 1.0e6,
        -np.sum(field_itrs_t * up, axis=1) * 1.0e6,
    )


def main() -> None:
    earth = simIncludeGravBody.gravBodyFactory().createEarth()
    circular_speed_m_s = np.sqrt(earth.mu / ORBIT_RADIUS_M)
    orbital_period_s = 2.0 * np.pi * np.sqrt(ORBIT_RADIUS_M**3 / earth.mu)
    duration_s = float(np.floor(orbital_period_s / 10.0) * 10.0)

    history = run_truth_environment(
        SpacecraftConfig(
            mass_kg=0.25,
            inertia_kg_m2=np.diag([8.0e-4, 9.0e-4, 7.0e-4]),
        ),
        InitialState(
            position_m=np.array([ORBIT_RADIUS_M, 0.0, 0.0]),
            velocity_m_s=np.array([0.0, circular_speed_m_s, 0.0]),
            mrp_body_from_inertial=np.zeros(3),
            angular_velocity_body_rad_s=np.zeros(3),
        ),
        SimulationConfig(duration_s=duration_s, step_s=10.0),
        TruthEnvironmentConfig(epoch_utc="2025-01-01T00:00:00", gravity_degree=20),
    )

    time_min = history.time_s / 60.0
    position_km = history.position_gcrs_m / 1000.0
    illumination = history.illumination_fraction
    magnetic_magnitude_ut = np.linalg.norm(history.magnetic_field_itrs_t, axis=1) * 1.0e6
    magnetic_north, magnetic_east, magnetic_down = local_magnetic_components(
        history.magnetic_field_itrs_t,
        history.geodetic_latitude_deg,
        history.geodetic_longitude_deg,
    )

    figure = plt.figure(figsize=(17, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    orbit_ax = figure.add_subplot(grid[0, 0], projection="3d")
    ground_ax = figure.add_subplot(grid[0, 1])
    light_ax = figure.add_subplot(grid[0, 2])
    magnetic_ax = figure.add_subplot(grid[1, 0])
    torque_ax = figure.add_subplot(grid[1, 1])
    vectors_ax = figure.add_subplot(grid[1, 2])

    sphere_u = np.linspace(0.0, 2.0 * np.pi, 80)
    sphere_v = np.linspace(0.0, np.pi, 40)
    earth_radius_km = EARTH_RADIUS_M / 1000.0
    sphere_x = earth_radius_km * np.outer(np.cos(sphere_u), np.sin(sphere_v))
    sphere_y = earth_radius_km * np.outer(np.sin(sphere_u), np.sin(sphere_v))
    sphere_z = earth_radius_km * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    orbit_ax.plot_surface(sphere_x, sphere_y, sphere_z, color="#3b82c4", alpha=0.18, linewidth=0)
    orbit_ax.plot(*position_km.T, color="0.55", linewidth=0.8)
    orbit_points = orbit_ax.scatter(
        *position_km.T,
        c=illumination,
        cmap="cividis",
        norm=Normalize(0.0, 1.0),
        s=7,
        label="illumination",
    )
    sample_indices = np.linspace(0, len(time_min) - 1, 22, dtype=int)
    magnetic_direction = history.magnetic_field_gcrs_t[sample_indices]
    magnetic_direction /= np.linalg.norm(magnetic_direction, axis=1, keepdims=True)
    orbit_ax.quiver(
        position_km[sample_indices, 0],
        position_km[sample_indices, 1],
        position_km[sample_indices, 2],
        magnetic_direction[:, 0],
        magnetic_direction[:, 1],
        magnetic_direction[:, 2],
        length=800.0,
        normalize=True,
        color="#9c36b5",
        linewidth=0.9,
        label="magnetic direction",
    )
    sun_direction = history.sun_direction_gcrs[0]
    orbit_ax.quiver(
        0.0,
        0.0,
        0.0,
        *sun_direction,
        length=10_000.0,
        normalize=True,
        color="#f59f00",
        linewidth=2.5,
        label="Sun direction",
    )
    axis_limit = 8_500.0
    orbit_ax.set(
        xlim=(-axis_limit, axis_limit),
        ylim=(-axis_limit, axis_limit),
        zlim=(-axis_limit, axis_limit),
    )
    orbit_ax.set_box_aspect((1, 1, 1))
    orbit_ax.set_xlabel("GCRS X [km]")
    orbit_ax.set_ylabel("GCRS Y [km]")
    orbit_ax.set_zlabel("GCRS Z [km]")
    orbit_ax.set_title("Orbit, sunlight, Sun and magnetic vectors")
    orbit_ax.legend(fontsize=7, loc="upper left")
    figure.colorbar(orbit_points, ax=orbit_ax, shrink=0.55, pad=0.08, label="Illumination")

    ground_scatter = ground_ax.scatter(
        history.geodetic_longitude_deg,
        history.geodetic_latitude_deg,
        c=illumination,
        cmap="cividis",
        norm=Normalize(0.0, 1.0),
        s=10,
    )
    ground_ax.set(xlim=(-180, 180), ylim=(-10, 10))
    ground_ax.set_xticks(np.arange(-180, 181, 60))
    ground_ax.grid(alpha=0.3)
    ground_ax.set_xlabel("East longitude [deg]")
    ground_ax.set_ylabel("Geodetic latitude [deg]")
    ground_ax.set_title(
        "Earth-fixed ground track\n"
        f"Altitude {history.geodetic_height_m.min() / 1000:.1f}–"
        f"{history.geodetic_height_m.max() / 1000:.1f} km"
    )
    figure.colorbar(ground_scatter, ax=ground_ax, label="Illumination")

    light_ax.fill_between(time_min, 0.0, illumination, color="#f9c74f", alpha=0.8)
    light_ax.plot(time_min, illumination, color="#9c6500", linewidth=1.2)
    light_ax.set(ylim=(-0.04, 1.04))
    light_ax.grid(alpha=0.25)
    light_ax.set_xlabel("Elapsed time [min]")
    light_ax.set_ylabel("Solar illumination fraction")
    light_ax.set_title("Basilisk Earth eclipse model")

    magnetic_ax.plot(time_min, magnetic_north, label="North")
    magnetic_ax.plot(time_min, magnetic_east, label="East")
    magnetic_ax.plot(time_min, magnetic_down, label="Down")
    magnetic_ax.plot(time_min, magnetic_magnitude_ut, color="black", linewidth=2, label="Magnitude")
    magnetic_ax.axhline(0.0, color="0.7", linewidth=0.8)
    magnetic_ax.grid(alpha=0.25)
    magnetic_ax.set_xlabel("Elapsed time [min]")
    magnetic_ax.set_ylabel("Magnetic field [µT]")
    magnetic_ax.set_title("IGRF-14 field at spacecraft")
    magnetic_ax.legend(ncol=2, fontsize=8)

    torque_pnm = history.gravity_gradient_torque_body_nm * 1.0e12
    torque_ax.plot(time_min, torque_pnm[:, 0], label="Body X")
    torque_ax.plot(time_min, torque_pnm[:, 1], label="Body Y")
    torque_ax.plot(time_min, torque_pnm[:, 2], label="Body Z")
    torque_ax.axhline(0.0, color="0.7", linewidth=0.8)
    torque_ax.grid(alpha=0.25)
    torque_ax.set_xlabel("Elapsed time [min]")
    torque_ax.set_ylabel("Torque [pN·m]")
    torque_ax.set_title("Gravity-gradient disturbance torque")
    torque_ax.legend(fontsize=8)

    field_body_unit = history.magnetic_field_body_t / np.linalg.norm(
        history.magnetic_field_body_t, axis=1, keepdims=True
    )
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for axis, color in enumerate(colors):
        vectors_ax.plot(
            time_min,
            history.sun_direction_body[:, axis],
            color=color,
            linewidth=1.8,
            label=f"Sun body {'XYZ'[axis]}",
        )
        vectors_ax.plot(
            time_min,
            field_body_unit[:, axis],
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"Mag body {'XYZ'[axis]}",
        )
    vectors_ax.axhline(0.0, color="0.7", linewidth=0.8)
    vectors_ax.set(ylim=(-1.05, 1.05))
    vectors_ax.grid(alpha=0.25)
    vectors_ax.set_xlabel("Elapsed time [min]")
    vectors_ax.set_ylabel("Body-frame unit-vector component")
    vectors_ax.set_title("Ideal ADS reference vectors")
    vectors_ax.legend(ncol=2, fontsize=7)

    eclipse_minutes = np.trapezoid(1.0 - illumination, history.time_s) / 60.0
    figure.suptitle(
        "PocketQube simulated truth environment — 550 km circular equatorial orbit\n"
        f"Epoch 2025-01-01 UTC · period {orbital_period_s / 60:.2f} min · "
        f"eclipse-equivalent duration {eclipse_minutes:.2f} min",
        fontsize=15,
    )
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "truth_environment_550km_equatorial_overview.png"
    figure.savefig(output_path, dpi=190)
    plt.close(figure)

    print(f"Saved {output_path.resolve()}")
    print(f"Orbital period: {orbital_period_s / 60.0:.3f} min")
    print(f"Eclipse-equivalent duration: {eclipse_minutes:.3f} min")
    print(
        "Magnetic magnitude range: "
        f"{magnetic_magnitude_ut.min():.3f} to {magnetic_magnitude_ut.max():.3f} µT"
    )
    print(
        "Gravity-gradient torque peak: "
        f"{np.linalg.norm(torque_pnm, axis=1).max():.3f} pN·m"
    )


if __name__ == "__main__":
    main()
