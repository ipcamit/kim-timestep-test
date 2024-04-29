from kim_python_utils.ase import CrystalGenomeTest
from numpy import multiply
from math import pi
from crystal_genome_util.aflow_util import get_stoich_reduced_list_from_prototype
from copy import deepcopy
from ase.build import bulk
import numpy as np
import sys
import subprocess
from ase import Atoms
from ase.io.lammpsdata import write_lammps_data

class MaximumStableLangevinTimestep(CrystalGenomeTest):
    """
    This test calculates the maximum stable Langevin timestep for a given model, as a proxy for the model stability. Additionally it calculates the stable timestep for a range of temperatures.
    """
    def _calculate(self, structure_index: int):
        """
        structure_index:
            KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). 
            This indicates which is being used for the current calculation.
        """
        # Define the timesteps and temperatures to test
        timesteps=np.array([0.000001, 0.00001, 0.0001, 0.001])
        temps=np.array([10.,100.,1000.,10000.])

        atoms = self.atoms[structure_index]
        species_list = list(set(atoms.get_chemical_symbols()))

        # Write the LAMMPS data file for the crystal
        crystal_lammps_file = f"{str(atoms.get_chemical_formula())}.dat"
        write_lammps_data(crystal_lammps_file, atoms, species_list)        

        stable_times_all = []

        # Loop over the temperatures
        for temp in temps:
            # Generate the LAMMPS input file template
            lammps_file_template = self.generate_lammps_file(atoms, species_list, timesteps) 
            lammps_file_name = f"{str(atoms.get_chemical_formula())}.lammps"
            with open(lammps_file_name,"w") as lmp_f:
                lmp_f.write(lammps_file_template)

            # Run LAMMPS for each temperature
            try:
                result = subprocess.check_output(["lmp", "-in", lammps_file_name, "-var", "temp_sim", str(temp)],stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Failed to run LAMMPS {e}")
                result = e.output

            # Parse the result to get the stable times
            if result is not None:
                stable_times = self.parse_lammps_result(result, f"{str(atoms.get_chemical_formula())}_{temp}_temp.dat",  temp, timesteps)
                stable_times_all.extend(stable_times)

        # Get the maximum stable time
        stable_times_all = np.array(stable_times_all)
        max_stable_time = np.max(stable_times_all[:,1])
        
        # Add the results to the property instance
        self._add_property_instance("maximum-stable-langevin-timestep")
        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index,write_stress=False,write_temp=False) # last two default to False
        self._add_key_to_current_property_instance("maximum-stable-timestep",max_stable_time, "ps")
        self._add_key_to_current_property_instance("all-stable-timestep-per-temperature",stable_times_all)

    def generate_lammps_file(self,
                        atoms:Atoms,
                        species_list:list,
                        timesteps:np.ndarray, 
                        n_steps:int = 100,
                        repl:int = 6) -> str:
        """
        Generate the LAMMPS input file for the test
        """
        file_str = f"# LAMMPS input script for testing simulation stability \n"
        file_str += f"# for the {self.model_name} model\n"
        file_str += f"\n\n"
        file_str += f"kim init {self.model_name}  metal\n\n"

        # read datafile
        file_str += f"read_data {str(atoms.get_chemical_formula())}.dat\n\n"

        file_str += f"kim interactions {' '.join(species_list)}\n"
        for i, s in enumerate(species_list):
            file_str += f"mass {i+1} {atoms.get_masses()[atoms.get_chemical_symbols().index(s)]}\n"

        file_str += "neighbor 1.0 bin\n"
        file_str += "neigh_modify every 1 delay 0 check no one 1000000 page 10000000\n\n"

        total_atoms = atoms.get_global_number_of_atoms()
        n_atoms = 250
        # ~ 250 atoms is the aim right now
        repl = int((n_atoms/total_atoms)**(1/3))
        if repl < 1:
            repl = 1

        file_str += f"replicate {repl} {repl} {repl}\n\n"

        file_str += f"variable temp equal temp\n"
        file_str += f"fix 1 all ave/time 1 1 1 v_temp ave one file {atoms.get_chemical_formula()}_${{temp_sim}}_temp.dat\n"

        file_str += "velocity all create ${temp_sim} 87287 rot yes dist gaussian\n"

        file_str += f"fix 2 all nve\n"
        # langevin dynamics with a damping constant of 10 fs. The damping is scaled by the average atomic mass of the system.
        # This provides a reasonable damping for a wide range of systems.
        langevin_damp = 10.0
        file_str += f"fix 3 all langevin ${{temp_sim}} ${{temp_sim}}  {langevin_damp * np.mean(atoms.get_atomic_numbers())} 1234567\n\n"
        file_str += f"thermo 1\n"

        for i, t in enumerate(timesteps):
            file_str += f"timestep {t}\n"
            file_str += f"run {n_steps}\n\n"

            file_str += f"print \"<<SUCCESS>> <<{t}>>\"\n" # success

        return file_str

    def parse_lammps_result(self, output, temp_file, temp, time,  variance_tol=0.2, n_steps=100):
        """
        Parse the LAMMPS output to get the stable times
        """
        temp_arr = np.loadtxt(temp_file, skiprows=2)
        n_successful_simulations = temp_arr.shape[0] // n_steps
        stable_times = []
        for i in range(n_successful_simulations):
            t = np.mean(temp_arr[i*n_steps: (i+1)*n_steps,1])
            if abs(t-temp) < variance_tol * temp:
                stable_times.append([temp, time[i]])
            else:
                # print(f"Rejecting simulation as temperature variance too high for T = {temp} and time = {time[i]}, mean temp = {t}.")
                pass
        return stable_times

# Alternatively, for debugging, give it atoms object or a list of atoms objects
atoms = bulk('NaCl','rocksalt',a=4.58)
test = MaximumStableLangevinTimestep(model_name="LennardJones612_UniversalShifted__MO_959249795837_003", atoms=atoms)
test()