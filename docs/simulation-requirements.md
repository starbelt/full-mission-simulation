# Simulation requirements

## Mission objective

Evaluate whether a PocketQube attitude determination and control system (ADCS) can point a selected body axis at a specified point on Earth using closed-loop orientation control.

## Primary capability

Given a target defined by geodetic latitude, longitude, and altitude, the simulation shall:

1. propagate the spacecraft orbit and true attitude;
2. compute the target line-of-sight in an inertial frame and transform it into the body frame;
3. simulate three-axis magnetometer, three-axis gyroscope, and sun-sensor measurements;
4. estimate spacecraft attitude and gyro bias from those measurements;
5. generate an attitude reference that points a configurable spacecraft body axis toward the target;
6. compute control commands and apply a configurable actuator model;
7. report pointing error, settling time, stability, target visibility, and actuator effort.

## State definitions

### Truth state

- simulation time
- position and velocity in an Earth-centered inertial frame
- body attitude quaternion relative to the inertial frame
- body angular velocity
- actuator state

### Estimated state

- attitude quaternion
- gyro bias
- attitude covariance or equivalent uncertainty measure
- optional position and velocity estimate

## Environment models

Initial models should include:

- Earth rotation and transformations among inertial, Earth-fixed, local orbital, and body frames
- spherical Earth target geometry and Earth occlusion
- two-body orbital propagation, with perturbations added later
- Sun ephemeris and eclipse detection
- global geomagnetic field evaluated at time and geodetic position
- configurable sensor noise, bias, scale-factor error, misalignment, sampling rate, and dropout

## Geomagnetic orbit characterization

The geomagnetic field can support position estimation by comparing magnetometer measurements with a global field model. A single magnetic-field vector does not generally identify a unique three-dimensional position. Position observability depends on combining a time history of field measurements with:

- attitude knowledge or simultaneous attitude estimation;
- orbital dynamics and a prior state/covariance;
- accurate measurement time;
- Sun-sensor measurements when available; and
- preferably an independent position observation or initial orbit estimate.

The simulator should therefore treat magnetic orbit characterization as model-aided navigation rather than a direct magnetic lookup. Candidate estimators include an extended or unscented Kalman filter whose state contains position, velocity, attitude, and sensor biases.

## Development stages

1. **Frames and geometry:** frame transformations, quaternions, geodetic targets, line-of-sight, and visibility.
2. **Truth simulation:** orbit and rigid-body attitude propagation.
3. **Environment:** Sun vector, eclipse state, and global geomagnetic field.
4. **Sensors:** gyro, magnetometer, and sun-sensor measurement models.
5. **Attitude determination:** initial vector-based solution followed by a recursive estimator.
6. **Pointing guidance and control:** target attitude generation, controller, and actuator dynamics.
7. **Navigation experiment:** augment the estimator with position and velocity and quantify magnetic-map observability.
8. **Monte Carlo evaluation:** sensor, model, initial-state, and disturbance uncertainty sweeps.

## Principal outputs

- inertial and Earth-fixed trajectory
- true and estimated attitude
- target visibility intervals
- boresight-to-target angular error
- attitude-estimation error
- position-estimation error when navigation is enabled
- covariance consistency
- actuator commands, saturation, and accumulated effort
- Monte Carlo summary statistics
