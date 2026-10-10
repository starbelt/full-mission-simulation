"""Package for the PC-first closed-loop ADCS prototype.

Layering, strictly one-directional:

``rotors``            TGR 29-12 rotor configuration and 3-wheel set
``wheel_board``       board stand-in: executes commands, owns wheel limits
``quaternions``       attitude kinematics (scalar-first, ``C_BN``)
``estimator``         onboard-like ADS: gyro propagation + absolute correction
``controller``        quaternion / body-rate feedback controller + allocation
``plant``             truth rigid body + rotors (Basilisk-backed)
``sensor_source``     synthetic sensor measurements derived from the plant
``slewos``            read-only adapter to the sibling SlewOS (FA01) planner
``closed_loop``       wiring of the above into runs
"""
