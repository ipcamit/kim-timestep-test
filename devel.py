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
    def _calculate(self, structure_index: int):
        """
        structure_index:
            KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). 
            This indicates which is being used for the current calculation.
        """
        times=np.array([0.000001, 0.00001, 0.0001, 0.001])
        temps=np.array([10.,100.,1000.,10000.])
        # f = open("debug.txt", "w")
        f = sys.stdout
        atoms = self.atoms[structure_index]

        species_list = list(set(atoms.get_chemical_symbols()))

        crystal_lammps_file = f"{str(atoms.get_chemical_formula())}.lammps"
        write_lammps_data(crystal_lammps_file, atoms, species_list)
        f.write(f"Written lammps data file: {crystal_lammps_file}\n")
        

        stable_times_all = []
        for temp in temps:
            lammps_file = self.generate_lammps_file(atoms, species_list, temp, times) 
            lammps_file_name = f"{str(atoms.get_chemical_formula())}_{temp}.lammps"
            with open(lammps_file_name,"w") as lmp_f:
                lmp_f.write(lammps_file)
            try:
                result = subprocess.check_output(["lmp", "-in", lammps_file_name],stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Failed to run LAMMPS {e}")
                result = e.output

            f.write(f"SIM: file {lammps_file_name}\n")

            # Parse the result
            if result is not None:
                stable_times = self.parse_lammps_result(result, f"{str(atoms.get_chemical_formula())}_{temp}_temp.dat",  temp, times)
                stable_times_all.extend(stable_times)
            f.write(f"Stable times for temperature {temp}: {stable_times}\n")

        stable_times_all = np.array(stable_times_all)
        f.write(f"Stable times: {stable_times_all}\n")
        max_stable_time = np.max(stable_times_all[:,1])
        
        # self._add_property_instance("binding-energy-relation-crystal")
        self._add_property_instance("maximum-stable-langevin-timestep")

        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index,write_stress=False,write_temp=False) # last two default to False

        # self._add_key_to_current_property_instance("average-wigner-seitz-radius",average_wigner_seitz_radius,"angstrom")
        print("Maximum stable timestep: ", max_stable_time)

        self._add_key_to_current_property_instance("maximum-stable-timestep",max_stable_time, "ps")

        print("Maximum stable timestep: ", max_stable_time)

        self._add_key_to_current_property_instance("maximum-stable-timestep-per-temperature",stable_times_all)
        # with open("debug"+str(structure_index)+".txt","w") as f:
        #     for r,u in zip(average_wigner_seitz_radius,binding_potential_energy_per_atom):
        #         f.write(str(r)+" "+str(u)+"\n")

    def generate_lammps_file(self,
                        atoms:Atoms,
                        species_list:list,
                        temp:float,
                        times:np.ndarray, 
                        nsteps:int = 100,
                        repl:int = 6) -> str:

        file_str = f"# LAMMPS input script for testing simulation stability \n"
        file_str += f"# for the {self.model_name} model\n"
        file_str += f"\n\n"
        file_str += f"kim init {self.model_name}  metal\n\n"

        # read datafile
        file_str += f"read_data {str(atoms.get_chemical_formula())}.lammps\n\n"

        file_str += f"kim interactions {' '.join(species_list)}\n"
        for i, s in enumerate(species_list):
            file_str += f"mass {i+1} {atoms.get_masses()[atoms.get_chemical_symbols().index(s)]}\n"

        file_str += "neighbor 1.0 bin\n"
        file_str += "neigh_modify every 1 delay 0 check no one 1000000 page 10000000\n\n"

        n_atoms = atoms.get_global_number_of_atoms()
        # ~ 250 atoms is the aim right now
        repl = int((250/n_atoms)**(1/3))
        if repl < 1:
            repl = 1

        file_str += f"replicate {repl} {repl} {repl}\n\n"

        file_str += f"variable temp equal temp\n"
        file_str += f"fix 2 all ave/time 1 1 1 v_temp ave one file {atoms.get_chemical_formula()}_{temp}_temp.dat\n"

        file_str += f"velocity all create {temp} 87287 rot yes dist gaussian\n"

        # TODO: Use Langevin thermostat
        file_str += f"fix 1 all nvt temp {temp} {temp} $(1.0*dt)\n\n"
        file_str += f"thermo 1\n"

        for i, t in enumerate(times):
            file_str += f"timestep {t}\n"
            file_str += f"run {nsteps}\n\n"

            file_str += f"print \"<<SUCCESS>> <<{t}>>\"\n" # success

        return file_str

    def parse_lammps_result(self, output, temp_file, temp, time,  variance_tol=0.5, nsteps=100):
        temp_arr = np.loadtxt(temp_file, skiprows=2)
        n_successful_simulations = temp_arr.shape[0] // nsteps
        print(n_successful_simulations,temp_arr.shape[0], nsteps)
        stable_times = []
        for i in range(n_successful_simulations):
            t = np.mean(temp_arr[i*nsteps: (i+1)*nsteps,1])
            print(t, temp, variance_tol * temp)
            if abs(t-temp) < variance_tol * temp:
                stable_times.append([temp, time[i]])
            else:
                print(f"Rejecting simulation as temperature variance too high for T = {temp} and time = {time[i]}, mean temp = {t}.")
        return stable_times

# This queries for equilibrium structures in this prototype and builds atoms
# test = BindingEnergyVsWignerSeitzRadius(model_name="MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002", stoichiometric_species=['Fe','P'], prototype_label='AB_oP8_62_c_c')
                
# Alternatively, for debugging, give it atoms object or a list of atoms objects
atoms = bulk('NaCl','rocksalt',a=4.58)
test = MaximumStableLangevinTimestep(model_name="LennardJones612_UniversalShifted__MO_959249795837_003", atoms=atoms)
test()