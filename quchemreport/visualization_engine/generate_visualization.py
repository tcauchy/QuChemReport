from collections import defaultdict
import json
import sys
from quchemreport.config.config import Config
from quchemreport.data_enrichment.compute_flags import ComputeFlags
from quchemreport.utility_services.computed_data import ComputedData
from quchemreport.visualization_engine import visu_mayavi, visu_plots, visu_txt


def generate_molecule_representation(
    computed_data: ComputedData, 
    nuclear_repulsion_energy_from_xyz, elements_3D_coords, atom_pairs, atoms_Z,
    log_idx, sources="N/A"
):
    
    output_dir = computed_data.metadata.output_dir
    
    if (
        nuclear_repulsion_energy_from_xyz is not None
        and elements_3D_coords is not None
        and atom_pairs is not None
        and atoms_Z is not None
    ):
        visu_mayavi.topoWithPCA(
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            file_name=output_dir+f"img-log{log_idx+1}"
        )
        print(f"Picture generated for one topology (Source: {sources})")

def output_molecule_representations(computed_data: ComputedData):
    log_files = computed_data.metadata.log_files

    all_nuclear_repulsion_energy_from_xyz = computed_data.results.geometry.nuclear_repulsion_energy_from_xyz
    all_elements_3D_coords = computed_data.results.geometry.elements_3D_coords
    all_atom_pairs = computed_data.molecule.connectivity.atom_pairs
    all_atoms_Z = computed_data.molecule.atoms_Z

    # Generate an image by groups of topology
    groups = computed_data.derived.molecule.topology_groups
    for group in groups:
        
        # Skip if no data
        if group is None:
            continue
        
        # Idx of the base log of the group
        base_log_idx = group[0]

        nuclear_repulsion_energy_from_xyz = all_nuclear_repulsion_energy_from_xyz[base_log_idx]
        elements_3D_coords = all_elements_3D_coords[base_log_idx]
        atom_pairs = all_atom_pairs[base_log_idx]
        atoms_Z = all_atoms_Z[base_log_idx]

        sources = ", ".join([log_files[i] for i in group])
        generate_molecule_representation(
            computed_data,
            nuclear_repulsion_energy_from_xyz,
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            base_log_idx,
            sources=sources
        )


def output_MO_diagrams_clean(computed_data: ComputedData):
    
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    
    MO_data = computed_data.derived.wavefunction.MO_analysis.MO_diagrams
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    if (
        MO_data.alpha is not None
        or MO_data.beta is not None
        or MO_data.restricted is not None
    ):
        print("\nGenerating MO Diagrams ...")
    
    # Generate visualization of the MO Diagrams for unrestricted calculation
    if MO_data.alpha is not None:
            visu_mayavi.viz_MO(
                MO_data.alpha.discretized_MO_voxels, 
                MO_data.alpha.grid_X, 
                MO_data.alpha.grid_Y, 
                MO_data.alpha.grid_Z,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                file_name=output_dir+f"img-log{ref_id}", 
                labels=MO_data.alpha.labels
            )
            print("Generation of the image for Alpha MO Diagram done")
            
    if MO_data.beta is not None:
        visu_mayavi.viz_MO(
            MO_data.beta.discretized_MO_voxels, 
            MO_data.beta.grid_X, 
            MO_data.beta.grid_Y, 
            MO_data.beta.grid_Z,
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            file_name=output_dir+f"img-log{ref_id}", 
            labels=MO_data.beta.labels
        )
        print("Generation of the image for Beta MO Diagram done")
    
    # Generate visualization of the MO Diagram for restricted calculation
    if MO_data.restricted is not None:
        visu_mayavi.viz_MO(
            MO_data.restricted.discretized_MO_voxels, 
            MO_data.restricted.grid_X, 
            MO_data.restricted.grid_Y, 
            MO_data.restricted.grid_Z, 
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            file_name=output_dir+f"img-log{ref_id}", 
            labels=MO_data.restricted.labels
        )
        print("Generation of the image for restricted MO Diagram done")


def output_MEP_maps(computed_data: ComputedData):
    
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    # Skip if data is missing
    if (
        elements_3D_coords is None
        or atom_pairs is None
        or atoms_Z is None
    ):
        return
    
    rho = computed_data.derived.wavefunction.MEP_maps.rho_voxels
    MEP = computed_data.derived.wavefunction.MEP_maps.MEP_voxels
    grid_X = computed_data.derived.wavefunction.MEP_maps.grid_X
    grid_Y = computed_data.derived.wavefunction.MEP_maps.grid_Y
    grid_Z = computed_data.derived.wavefunction.MEP_maps.grid_Z
    
    if (
        rho is not None
        and MEP is not None
        and grid_X is not None
        and grid_Y is not None
        and grid_Z is not None
    ):
        print("\nGenerating MEP Map ...")
        visu_mayavi.viz_Potential(
            rho,
            MEP,
            grid_X,
            grid_Y,
            grid_Z,
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            file_name=output_dir+f"img-log{ref_id}"
        )
        print("Generation of the MEP Map image done")


def output_fukui_functions(computed_data: ComputedData):
    
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    # Skip if data is missing
    if (
        elements_3D_coords is None
        or atom_pairs is None
        or atoms_Z is None
    ):
        return
    
    print("")
    
    # Generate the output of the fukui functions for the SP Plus if the data is available
    f_plus  = computed_data.derived.reactivity.fukui_functions.SP_plus
    if f_plus is not None:
        
        delta_rho_SPp = f_plus.rho_voxels
        grid_X_SPp = f_plus.grid_X
        grid_Y_SPp = f_plus.grid_Y
        grid_Z_SPp = f_plus.grid_X
        
        if (
            delta_rho_SPp is not None
            and grid_X_SPp is not None
            and grid_Y_SPp is not None
            and grid_Z_SPp is not None
        ):
            print("Generating the image for Fukui SP Plus ...")
            visu_mayavi.viz_Fukui(
                delta_rho_SPp,
                grid_X_SPp,
                grid_Y_SPp,
                grid_Z_SPp,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                file_name=output_dir+"img",
                labels="SP_plus"
            )
            print("Generation of the image for Fukui SP Plus done")
    
    # Generate the output of the fukui functions for the SP Minus if the data is available
    f_minus = computed_data.derived.reactivity.fukui_functions.SP_minus
    if f_minus is not None:
        
        delta_rho_SPm = f_minus.rho_voxels
        grid_X_SPm = f_minus.grid_X
        grid_Y_SPm = f_minus.grid_Y
        grid_Z_SPm = f_minus.grid_Z
        
        if (
            delta_rho_SPm is not None
            and grid_X_SPm is not None
            and grid_Y_SPm is not None
            and grid_Z_SPm is not None
        ):
            print("Generating the image for Fukui SP Minus ...")
            visu_mayavi.viz_Fukui(
                delta_rho_SPm,
                grid_X_SPm,
                grid_Y_SPm,
                grid_Z_SPm,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                file_name=output_dir+"img",
                labels="SP_minus"
            )
            print("Generation of the image for Fukui SP Minus done")
    
    # Generate the output of the fukui functions for the Dual if the data is available
    f_dual = computed_data.derived.reactivity.fukui_functions.dual
    if f_dual is not None:
        
        delta_rho_dual = f_dual.grid_X
        grid_X_dual = f_dual.grid_X
        grid_Y_dual = f_dual.grid_Y
        grid_Z_dual = f_dual.grid_Z
        
        if (
            delta_rho_dual is not None
            and grid_X_dual is not None
            and grid_Y_dual is not None
            and grid_Z_dual is not None
        ):
            print("Generating the image for Fukui Dual ...")
            visu_mayavi.viz_Fdual(
                delta_rho_dual,
                grid_X_dual,
                grid_Y_dual,
                grid_Z_dual,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                file_name=output_dir+"img"
            )
            print("Generation of the image for Fukui Dual done")


def output_es_absorption_and_CD_spectrum(computed_data: ComputedData, do_cd, do_abso):
    
    output_dir = computed_data.metadata.output_dir
    log_files = computed_data.metadata.log_files
    
    print("")
    
    for log_idx, log_file in enumerate(log_files):
    
        et_energies = computed_data.results.excited_states.et_energies[log_idx]
        et_rotats = computed_data.results.excited_states.et_rot[log_idx]
        et_oscs = computed_data.results.excited_states.et_oscs[log_idx]
        
        abso_spectrum = computed_data.derived.excited_states.abso_spectrum[log_idx]
        CD_spectrum = computed_data.derived.excited_states.CD_spectrum[log_idx]
        xvalues = computed_data.derived.excited_states.xvalues[log_idx]
        
        # Skip if no data
        if (
            abso_spectrum is None
            or len(abso_spectrum) == 0
            or xvalues is None
            or et_energies is None
        ):
            continue
    
        
        # Output calculated CD spectra in text files and figures
        if (
            do_cd 
            and CD_spectrum is not None 
            and len(CD_spectrum) != 0
            and et_rotats is not None
        ):
            print(f"Generating Circular dichroism spectrum done for {log_file} ...")
            # Write both Absorption and Circular dichroism spectra in text file to compare with experimental data
            visu_txt.textfile_CD(xvalues, abso_spectrum, CD_spectrum, file_name=output_dir+f"log{log_idx + 1}-",)                        
            # IF CD plot CD spectrum 
            visu_plots.absoCD(et_energies, et_rotats, xvalues, CD_spectrum, file_name=output_dir+"img"+f"-log{log_idx + 1}",)
            print(f"Circular dichroism spectrum done for {log_file}.")
        
        # Output calculated Absorption spectra in text files and figures
        if (
            do_abso
            and et_oscs is not None
        ):
            print(f"Generating Absorption spectrum done for {log_file}.")
            # If no CD write only Absorption spectrum in text file to compare with experimental data
            if (
                not do_cd 
                or len(CD_spectrum) == 0
            ):
                visu_txt.textfile_UV(xvalues, abso_spectrum, file_name=output_dir+f"log{log_idx + 1}-")       

            # Plotting UV spectrum 
            visu_plots.absoUV(et_energies, et_oscs, xvalues, abso_spectrum, file_name=output_dir+"img"+f"-log{log_idx + 1}",)
            print(f"Absorption spectrum done for {log_file}.")


def output_es_EDD_OIF_DIP(computed_data: ComputedData):
    
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    log_files = computed_data.metadata.log_files
    
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    for log_idx, _ in enumerate(log_files):
    
        et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
        et_oscs = computed_data.results.excited_states.et_oscs[log_idx]
        et_rotats = computed_data.results.excited_states.et_rot[log_idx]
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        
        TD_output = computed_data.derived.excited_states.TD_output[log_idx]
        grid_X = computed_data.derived.excited_states.grid_X[log_idx]
        grid_Y = computed_data.derived.excited_states.grid_Y[log_idx]
        grid_Z = computed_data.derived.excited_states.grid_Z[log_idx]
        all_data_dip = computed_data.derived.excited_states.all_data_dip[log_idx]
        Tozer_lambda = computed_data.derived.excited_states.Tozer_lambda[log_idx]
        
        # Skip if data is missing
        if (
            elements_3D_coords is None
            or atom_pairs is None
            or atoms_Z is None
            or et_transitions is None
            or et_oscs is None
            or et_rotats is None
            or et_sym is None
            or TD_output is None
            or grid_X is None
            or grid_Y is None
            or grid_Z is None
            or all_data_dip is None
            or Tozer_lambda is None
        ):
            continue
        
        # Skip if et_transitions is empty
        if (len(et_transitions) == 0):
            continue
        
        print("")
        
        for k, _ in enumerate(et_transitions):
            
            # Skip if no Tozer_lambda computed
            # Only print the visualization of the user-selected excited state
            if (
                Tozer_lambda[k] is None
            ):
                continue
            
            print("Excited State EDD visualization in progress for the transition:", k+1)
            discretized_EDD_voxels = [TD_output[k][0]]
            visu_mayavi.viz_EDD(
                discretized_EDD_voxels,
                grid_X,
                grid_Y,
                grid_Z,
                elements_3D_coords,
                atom_pairs,
                atoms_Z,
                et_sym[k],
                file_name=output_dir+"img"+f"-log{log_idx + 1}", labels=[k+1]
            ) 
            
            if (len(et_rotats) > 0) and (abs(et_rotats[k]) > 10.):
                    #calculate only the elect and magnetic dipole for chiral compounds
                    print("Generating overlap image for the selected transition:", k+1)
                    visu_mayavi.viz_Oif([TD_output[k][3]],
                    grid_X,
                    grid_Y,
                    grid_Z,
                    elements_3D_coords,
                    atom_pairs,
                    atoms_Z,
                    et_sym[k],
                    file_name=output_dir+"img"+f"-log{log_idx + 1}", labels=[k+1]
                )            

            data_dip = all_data_dip[k]
            if data_dip is not None:
                print("Generating transition dipole images for selected transition:", k+1),
                visu_mayavi.viz_dip(
                    data_dip,
                    elements_3D_coords,
                    atom_pairs,
                    atoms_Z,
                    et_sym[k],
                    file_name=output_dir+"img"+f"-log{log_idx + 1}",
                    labels=[k+1]
                )


def output_optimized_es_absorption_and_CD_spectrum(computed_data: ComputedData, config: Config, do_emi, do_CPL):
    
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    output_dir = computed_data.metadata.output_dir
    
    for log_idx,log_file in enumerate(log_files):
        
        jobs = job_types[log_idx]
        
        if not any("_es" in jb for jb in jobs):
            continue
        
        emi_state = config.logfiles[log_idx].excited_state_number
        if emi_state is None:
            # Invalid emi_state so it will stop when checking the emi_index
            emi_state = -1
        
        et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
        
        emi_index = emi_state - 1
        if (emi_index) < 0:
            print("Incoherent excited state optimization detected. Problem with root number")
            print(f"Emission state (user input) = {emi_state} ; expected to be within [1, {len(et_transitions)}]")
            continue
        
        et_energy = computed_data.results.excited_states.et_energies[log_idx]
        et_osc = computed_data.results.excited_states.et_oscs[log_idx]
        et_rot = computed_data.results.excited_states.et_rot[log_idx]
        
        # Skip if data is missing
        if (
            et_energy is None
            or et_osc is None
        ):
            continue
        
        emi_energy = et_energy[emi_index]
        emi_osc = et_osc[emi_index]
        
        if et_rot is not None:
            emi_rotat = et_rot[emi_index]
        else:
            emi_rotat = 0.0
        
        xvalues = computed_data.derived.optimized_excited_states.xvalues[log_idx]
        emi_spectrum = computed_data.derived.optimized_excited_states.emi_spectrum[log_idx]
        CPL_spectrum = computed_data.derived.optimized_excited_states.CPL_spectrum[log_idx]
        
        # Skip if no data
        if (
            emi_spectrum is None
            or len(emi_spectrum) == 0
            or xvalues is None
        ):
            continue
        
        if (
            do_emi
        ):
            if (
                do_CPL
                and CPL_spectrum is not None
                and len(CPL_spectrum) != 0
            ):
                # Write both Emission and CPL spectra in text file to compare with experimental data
                visu_txt.textfile_emiCD(xvalues, emi_spectrum, CPL_spectrum, file_name=output_dir+f"log{log_idx}") 
            # If no CPL write only Emission spectrum in text file to compare with experimental data
            else :
                visu_txt.textfile_emiUV(xvalues, emi_spectrum, file_name=output_dir+f"log{log_idx}")   
            
            # Plotting emiUV spectrum 
            visu_plots.emiUV([emi_energy], [emi_osc], xvalues, emi_spectrum, file_name=output_dir+f"img-log{log_idx}")
            print(f"Generating Emission spectrum done for {log_file}.")            
        
        if (
            do_CPL
            and CPL_spectrum is not None
            and len(CPL_spectrum) != 0
        ):
            # plot CPL spectrum 
            visu_plots.emiCD([emi_energy], [emi_rotat], xvalues, CPL_spectrum, file_name=output_dir+f"img-log{log_idx}")
            print(f"CPL spectrum done for {log_file}.") 


def output_optimized_es_EDD_DIP(computed_data: ComputedData, config: Config):
    
    ref_id = computed_data.metadata.ref_log_file_idx
    output_dir = computed_data.metadata.output_dir
    log_files = computed_data.metadata.log_files
    job_types = computed_data.comp_details.general.job_type
    
    elements_3D_coords = computed_data.results.geometry.elements_3D_coords[ref_id]
    atom_pairs = computed_data.molecule.connectivity.atom_pairs[ref_id]
    atoms_Z = computed_data.molecule.atoms_Z[ref_id]
    
    print("")
    
    for log_idx,log_file in enumerate(log_files):
        
        jobs = job_types[log_idx]
        
        if not any("_es" in jb for jb in jobs):
            continue
    
        et_transitions = computed_data.results.excited_states.et_transitions[log_idx]
        emi_state = config.logfiles[log_idx].excited_state_number
        
        # Skip if data is missing
        if (
            et_transitions is None
            or emi_state is None
        ):
            return
        
        emi_index = emi_state - 1
        if (
            emi_index < 0
            or emi_index >= len(et_transitions)
        ):
            return
    
        et_sym = computed_data.results.excited_states.et_sym[log_idx]
        et_rot = computed_data.results.excited_states.et_rot[log_idx]
        
    
        TD_output = computed_data.derived.optimized_excited_states.TD_output[log_idx]
        grid_X = computed_data.derived.optimized_excited_states.grid_X[log_idx]
        grid_Y = computed_data.derived.optimized_excited_states.grid_Y[log_idx]
        grid_Z = computed_data.derived.optimized_excited_states.grid_Z[log_idx]
        all_data_dip = computed_data.derived.optimized_excited_states.all_data_dip[log_idx]
        
        # Skip if data is missing
        if (
            elements_3D_coords is None
            or atom_pairs is None
            or atoms_Z is None
            or et_sym is None
            or TD_output is None
            or grid_X is None
            or grid_Y is None
            or grid_Z is None
            or all_data_dip is None
        ):
            continue
        
        
        if et_rot is not None:
            emi_rotat=et_rot[emi_index]
        else:
            emi_rotat = 0.0
        
        # Using TD_output[0] instead of the index of the transition as TD_output only contain the transition of index emi_index here
        discretized_EDD_voxels = [TD_output[0][0]]
        
        print(f"Optimized Excited State EDD visualization in progress for the transition S{emi_state}")
        visu_mayavi.viz_EDD(
            discretized_EDD_voxels,
            grid_X,
            grid_Y,
            grid_Z,
            elements_3D_coords,
            atom_pairs,
            atoms_Z,
            et_sym, 
            file_name=output_dir+f"img-log{log_idx}-emi", labels=[f"S{emi_state}"]
        ) 
        
        if (emi_rotat != 0.0):
            data_dip = all_data_dip[emi_state]
            if data_dip is not None:
                visu_mayavi.viz_dip(
                    data_dip,
                    elements_3D_coords,
                    atom_pairs,
                    atoms_Z,
                    et_sym,
                    file_name=output_dir+f"img-log{log_idx}-emi",
                    labels=[f"S{emi_state}"]
                )
                print(f"Generating transition dipole images for selected transition: {emi_state}")


def generate_visualization(computed_data: ComputedData, config: Config, compute_flags: ComputeFlags):
    
    print("\n=================================================================")
    print(" Visualization")
    print("=================================================================")
    
    # Generate molecule representation
    if compute_flags.compute_molecule_representations:
        output_molecule_representations(computed_data)
    
    # Generate the outputs for the MO Diagrams
    if compute_flags.compute_MO_diagrams:
        output_MO_diagrams_clean(computed_data)
    
    # Generate the outputs for the MEP maps
    if compute_flags.compute_MEP_maps :
        output_MEP_maps(computed_data)
    
    # Generate the outputs for the fukui functions
    if compute_flags.compute_fukui_functions :
        output_fukui_functions(computed_data)
    
    # Generate the output for the Absorption and CD spectrum of the excited states
    do_abso = compute_flags.compute_es_Abso_spectrum
    do_cd = compute_flags.compute_es_CD_spectrum
    if (do_abso or do_cd):
        output_es_absorption_and_CD_spectrum(computed_data, do_cd, do_abso)
    
    # Generate the outputs for the EDD, OIF and DIP of the excited states
    if compute_flags.compute_es_transitions_and_EDD :
        output_es_EDD_OIF_DIP(computed_data)
    
    # Generate the output for the Absorption and CD spectrum of the optimized excited states
    do_emi = compute_flags.compute_optimized_es_emi_spectrum
    do_CPL = compute_flags.compute_optimized_es_CPL_spectrum
    if (do_emi or do_CPL):
        output_optimized_es_absorption_and_CD_spectrum(computed_data, config, do_emi, do_CPL)
    
    # Generate the outputs for the transitions of the optimized excited states
    if compute_flags.compute_optimized_es_transitions_and_EDD :
        output_optimized_es_EDD_DIP(computed_data, config)
    