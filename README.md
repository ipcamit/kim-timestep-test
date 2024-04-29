Langevin MD Timestep Test
=========================

This test runs an overdamped Langevin dynamics simulation. The test quantity is then the average temperature of the system. The test passes if the average temperature is within a certain tolerance of the target temperature. 

The damping constant is determined by scaling the damping time by the mean atomic mass units of the simulation cell. This provides a reasonable damping constant for the simulation cell, as LAMMPS apply the damping constant in `fix langevin` to scale the momentum term as `(m/damping) v`. Therefore for  any simulation cell the damping will now effectively be scaled as `((m/mean_mass)/damping) v`, providing somewhat equal footing.

Other defaults and assumptions are listed below, and can be modified by searching for the `Parameter` in the `devel.py` file.

Defaults:

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| `timesteps` | 1e-6 to 1e-3 fs | timesteps to tests simulations on |
| `temps` | 1e1 to 1e4 K | target temperature of the simulation |
| `n_atoms` | 250 | The desired number of atoms in simulation cell, it influences the parameter `repl`, which sets the replicatio of the cell, such that total number of particles are ~250 |
| `n_steps` | 1000 | Number of steps to run the simulation for |
| `variance_tol` | 0.2 | Tolerance for the average temperature fluctuation, 0.2 means, 0.8 * temp < temps < 1.2 * temp, where temp is the requested temperature |
| `langevin_damp` | 10.0 | timesteps to avgerage over for Langevin damping constant |


## Property defination
This test calculates two properties namely, `maximum-stable-timestep`, and
  `all-stable-timestep-per-temperature`. The `maximum-stable-timestep` is the absolute largest timestep that can run the simulation stably, irrespective of the temperature. this serves as a scalar value for comparing the stability of the simulation. The `all-stable-timestep-per-temperature` lists the combinations of all temperatures and timestemps for which the simulaitons ran stably. This serves as a detailed information on the stability of the simulation.