"""Render a clean close-up of orbit lighting, Sun, and magnetic vectors."""

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

    position_km = history.position_gcrs_m / 1000.0
    illumination = history.illumination_fraction

    figure = plt.figure(figsize=(10, 10), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    sphere_u = np.linspace(0.0, 2.0 * np.pi, 120)
    sphere_v = np.linspace(0.0, np.pi, 60)
    earth_radius_km = EARTH_RADIUS_M / 1000.0
    sphere_x = earth_radius_km * np.outer(np.cos(sphere_u), np.sin(sphere_v))
    sphere_y = earth_radius_km * np.outer(np.sin(sphere_u), np.sin(sphere_v))
    sphere_z = earth_radius_km * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    axis.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color="#2878b5",
        alpha=0.32,
        linewidth=0,
        shade=True,
    )

    axis.plot(*position_km.T, color="0.5", linewidth=1.0, alpha=0.7)
    orbit_points = axis.scatter(
        *position_km.T,
        c=illumination,
        cmap="cividis",
        norm=Normalize(0.0, 1.0),
        s=14,
        depthshade=False,
        label="Orbit lighting",
    )

    sample_indices = np.linspace(0, len(history.time_s) - 1, 30, dtype=int)
    magnetic_direction = history.magnetic_field_gcrs_t[sample_indices].copy()
    magnetic_direction /= np.linalg.norm(magnetic_direction, axis=1, keepdims=True)
    axis.quiver(
        position_km[sample_indices, 0],
        position_km[sample_indices, 1],
        position_km[sample_indices, 2],
        magnetic_direction[:, 0],
        magnetic_direction[:, 1],
        magnetic_direction[:, 2],
        length=950.0,
        normalize=True,
        color="#d946ef",
        linewidth=1.25,
        arrow_length_ratio=0.22,
        label="IGRF-14 field direction",
    )

    sun_direction = history.sun_direction_gcrs[0]
    axis.quiver(
        0.0,
        0.0,
        0.0,
        *sun_direction,
        length=11_000.0,
        normalize=True,
        color="#f59f00",
        linewidth=3.5,
        arrow_length_ratio=0.12,
        label="Direction to Sun",
    )

    axis_limit = 8_600.0
    axis.set(
        xlim=(-axis_limit, axis_limit),
        ylim=(-axis_limit, axis_limit),
        zlim=(-axis_limit, axis_limit),
    )
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24.0, azim=-57.0)
    axis.set_axis_off()
    axis.set_title(
        "550 km equatorial orbit: sunlight and geomagnetic field\n"
        "IGRF-14 · JPL DE430/SPICE · epoch 2025-01-01 UTC",
        fontsize=15,
        pad=4,
    )
    axis.legend(loc="upper left", frameon=False, fontsize=10)
    colorbar = figure.colorbar(
        orbit_points,
        ax=axis,
        shrink=0.58,
        pad=0.0,
        aspect=24,
    )
    colorbar.set_label("Solar illumination fraction", fontsize=11)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "earth_orbit_sun_eclipse_magnetic_closeup.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
