"""Plot IGRF-14 magnetic-field truth over a 550 km equatorial orbit."""

from pathlib import Path

import matplotlib
import numpy as np
from Basilisk.utilities import simIncludeGravBody

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pocketadcs_sim.dynamics import InitialState, SimulationConfig, SpacecraftConfig  # noqa: E402
from pocketadcs_sim.truth_environment import (  # noqa: E402
    TruthEnvironmentConfig,
    run_truth_environment,
)

EARTH_EQUATORIAL_RADIUS_M = 6_378_137.0
ALTITUDE_M = 550_000.0
ORBIT_RADIUS_M = EARTH_EQUATORIAL_RADIUS_M + ALTITUDE_M


def main() -> None:
    earth = simIncludeGravBody.gravBodyFactory().createEarth()
    circular_speed_m_s = np.sqrt(earth.mu / ORBIT_RADIUS_M)
    orbital_period_s = 2.0 * np.pi * np.sqrt(ORBIT_RADIUS_M**3 / earth.mu)

    duration_s = float(np.floor(orbital_period_s / 10.0) * 10.0)

    history = run_truth_environment(
        SpacecraftConfig(
            mass_kg=0.25,
            inertia_kg_m2=np.diag([8.0e-4, 8.0e-4, 8.0e-4]),
        ),
        InitialState(
            position_m=np.array([ORBIT_RADIUS_M, 0.0, 0.0]),
            velocity_m_s=np.array([0.0, circular_speed_m_s, 0.0]),
            mrp_body_from_inertial=np.zeros(3),
            angular_velocity_body_rad_s=np.zeros(3),
        ),
        SimulationConfig(duration_s=duration_s, step_s=10.0),
        TruthEnvironmentConfig(
            epoch_utc="2025-01-01T00:00:00",
            gravity_degree=20,
        ),
    )

    latitude = np.radians(history.geodetic_latitude_deg)
    longitude = np.radians(history.geodetic_longitude_deg)
    field = history.magnetic_field_itrs_t
    east_axis = np.column_stack((-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)))
    north_axis = np.column_stack(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        )
    )
    up_axis = np.column_stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )
    field_east_ut = np.sum(field * east_axis, axis=1) * 1.0e6
    field_north_ut = np.sum(field * north_axis, axis=1) * 1.0e6
    field_down_ut = -np.sum(field * up_axis, axis=1) * 1.0e6
    field_magnitude_ut = np.linalg.norm(field, axis=1) * 1.0e6

    figure = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    orbit_ax = figure.add_subplot(1, 2, 1, projection="3d")
    components_ax = figure.add_subplot(1, 2, 2)

    sphere_u = np.linspace(0.0, 2.0 * np.pi, 80)
    sphere_v = np.linspace(0.0, np.pi, 40)
    earth_radius_km = EARTH_EQUATORIAL_RADIUS_M / 1000.0
    sphere_x = earth_radius_km * np.outer(np.cos(sphere_u), np.sin(sphere_v))
    sphere_y = earth_radius_km * np.outer(np.sin(sphere_u), np.sin(sphere_v))
    sphere_z = earth_radius_km * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    orbit_ax.plot_surface(sphere_x, sphere_y, sphere_z, color="#2878b5", alpha=0.18, linewidth=0)

    position_km = history.position_gcrs_m / 1000.0
    orbit_ax.plot(*position_km.T, color="#202020", linewidth=1.4, label="550 km orbit")
    arrow_indices = np.linspace(0, len(history.time_s) - 1, 28, dtype=int)
    arrow_directions = history.magnetic_field_gcrs_t[arrow_indices]
    arrow_directions /= np.linalg.norm(arrow_directions, axis=1, keepdims=True)
    arrow_length_km = 900.0
    orbit_ax.quiver(
        position_km[arrow_indices, 0],
        position_km[arrow_indices, 1],
        position_km[arrow_indices, 2],
        arrow_directions[:, 0],
        arrow_directions[:, 1],
        arrow_directions[:, 2],
        length=arrow_length_km,
        normalize=True,
        color="#d62728",
        linewidth=1.1,
        label="IGRF-14 direction (scaled)",
    )
    limit = 8_000.0
    orbit_ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    orbit_ax.set_box_aspect((1, 1, 1))
    orbit_ax.set_xlabel("GCRS X [km]")
    orbit_ax.set_ylabel("GCRS Y [km]")
    orbit_ax.set_zlabel("GCRS Z [km]")
    orbit_ax.set_title("Field direction along one equatorial orbit")
    orbit_ax.legend(loc="upper left", fontsize=8)

    time_minutes = history.time_s / 60.0
    components_ax.plot(time_minutes, field_north_ut, label="North", linewidth=1.5)
    components_ax.plot(time_minutes, field_east_ut, label="East", linewidth=1.5)
    components_ax.plot(time_minutes, field_down_ut, label="Down", linewidth=1.5)
    components_ax.plot(
        time_minutes,
        field_magnitude_ut,
        label="Magnitude",
        color="black",
        linewidth=2.0,
    )
    components_ax.axhline(0.0, color="0.7", linewidth=0.8)
    components_ax.grid(alpha=0.25)
    components_ax.set_xlabel("Elapsed time [min]")
    components_ax.set_ylabel("Magnetic field [µT]")
    components_ax.set_title("Local field components along the orbit")
    components_ax.legend(ncol=2)

    figure.suptitle(
        "PocketQube magnetic truth — 550 km circular equatorial orbit\n"
        "IGRF-14, epoch 2025-01-01; GGM03S degree 20 propagation",
        fontsize=13,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "magnetic_field_550km_equatorial_orbit.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    print(f"Saved {output_path.resolve()}")
    print(f"Circular speed: {circular_speed_m_s:.3f} m/s")
    print(f"Orbital period: {orbital_period_s / 60.0:.3f} min")
    print(
        "Field magnitude range: "
        f"{field_magnitude_ut.min():.3f} to {field_magnitude_ut.max():.3f} µT"
    )


if __name__ == "__main__":
    main()
