import os
import sys
import traceback
import json
import pickle
import hashlib
import cclib
import openbabel.pybel as pybel
import numpy as np
import openbabel as ob
import sklearn.preprocessing

from quchemreport.config.config import Config
from quchemreport.utility_services.log_data import LogData

# constants
CstBohr2Ang = 0.52917721092
CstHartree2eV = 27.21138505
CstHartree2cm1 = 219474.6313708
scanlog_version = "1.0.2"

""" Scanlog Exception class.
"""
class ScanlogException(Exception):
    pass


def split_logfile_gaussian(logfile, log_storage_path, verbose):
    log_files = []
    
    TERM = "Normal termination"
    
    with open(logfile, 'r') as log_fd:
        lines = log_fd.readlines()
    nbl = len(lines)
    
    file_cnt = 0
    base_fname = os.path.basename(logfile).rsplit('.', 1)[0]
    log_pat = os.path.join(log_storage_path, "%s_step_%s.log" % (base_fname, "%d"))
    cur_log = log_pat % file_cnt
    
    if verbose:
        print(">>> Processing", cur_log, "...")
        
    # FLAG to add copyright at the beginning of each step.
    cur_log_fd = open(cur_log, "w")
    file_start = True
    for cur_l, line in enumerate(lines):
        if file_start:
            cur_log_fd.write(" Copyright (c) 1988,1990,1992,1993,1995,1998,2003,2009,2013,\n")
            cur_log_fd.write("            Gaussian, Inc.  All Rights Reserved.\n")
            file_start = False
            
        cur_log_fd.write(line)
        if line.find(TERM) > -1 :
            file_start = True
            if verbose:
                print("=> ",line)
            log_files.append(cur_log)
            cur_log_fd.close()
            file_cnt += 1
            if nbl > (cur_l + 1) :
                cur_log = log_pat % file_cnt
                if verbose:
                    print(">>> Processing", cur_log, "...")
                cur_log_fd = open(cur_log, "w")
                
    if not cur_log_fd.closed :
        cur_log_fd.close()
        
    return log_files


def split_logfile_gamess(logfile, log_storage_path, verbose):
    log_files = []
    
    TERM = "TERMINATED NORMALLY"
    
    with open(logfile, 'r') as log_fd:
        lines = log_fd.readlines()
        
    for line in lines:
        if line.find(TERM) > -1 :
            if verbose:
                print("=> ",line)
            log_files.append(logfile)
            
    return log_files


def split_logfile_orca(logfile, log_storage_path, verbose):
    log_files = []
    
    TERM = "Timings for individual modules:"
    program_version = ""
    
    with open(logfile, 'r') as log_fd:
        lines = log_fd.readlines()
    nbl = len(lines)

    file_cnt = 0
    base_fname = os.path.basename(logfile).rsplit('.', 1)[0]
    log_pat = os.path.join(log_storage_path, "%s_step_%s.log" % (base_fname, "%d"))
    cur_log = log_pat % file_cnt
    
    if verbose:
        print(">>> Processing", cur_log, "...")
        
    # FLAG to add software name and version at the beginning of each step.
    cur_log_fd = open(cur_log, "w")
    file_start = True
    for cur_l, line in enumerate(lines):
        if file_start:
            cur_log_fd.write("\nO   R   C   A\n")
            if program_version != "" :
                cur_log_fd.write("\n"+program_version+"\n")
            file_start = False

        if "Program Version" in line:
            program_version = line

        cur_log_fd.write(line)
        if line.find(TERM) > -1 :
            file_start = True
            if verbose:
                print("=> ",line)
            log_files.append(cur_log)
            cur_log_fd.close()
            file_cnt += 1
            if nbl > (cur_l + 1) :
                cur_log = log_pat % file_cnt
                if verbose:
                    print(">>> Processing", cur_log, "...")
                cur_log_fd = open(cur_log, "w")
                
    if not cur_log_fd.closed :
        cur_log_fd.close()
        
    return log_files


def split_logfile(logfile, solver, log_storage_path="", verbose=False):
    try:
        if verbose:
            print(">>> SOLVER:", solver)

        def unsupported_solver(*args, **kwargs):
            raise ScanlogException(f"Unsupported solver : {solver}.")

        dispatch = {
            "gaussian": split_logfile_gaussian,
            ### TODO: GAMESS not tested, accepted here only for Riken
            ### DB insertion purpose (only OPT mono step)
            "gamess": split_logfile_gamess,
            ### TODO: NWChem not tested
            ### we assume normal termination already tested
            "nwchem": lambda *_: [logfile],
            "orca": split_logfile_orca,
        }

        handler = dispatch.get(solver, unsupported_solver)
        log_files = handler(logfile, log_storage_path, verbose)

        if verbose:
            print(">>> Steps :", log_files, "\n")

        return log_files

    except ScanlogException as err:
        raise err
    except Exception:
        if verbose:
            traceback.print_exc()
        raise ScanlogException("File splitting failed.")


"""Redefining nuclear_repulsion_energy with 5 decimals of precision on coords.
"""
def nuclear_repulsion_energy(data, slice_id=-1):
    nre = 0.0
    for i in range(data.natom):
        ri = np.array([float("%.5f" % k) for k in data.atomcoords[slice_id][i]])
        zi = data.atomnos[i]
        for j in range(i + 1, data.natom):
            rj = np.array([float("%.5f" % k) for k in data.atomcoords[slice_id][j]])
            zj = data.atomnos[j]
            d = np.linalg.norm(ri - rj)
            nre += zi * zj / d
    return float("%.5f" % (nre * CstBohr2Ang))


"""Utility function to simplify data recording from dict or other object.
"""
def _try_key_insertion(res_json, key, obj, obj_path=[], nullable=True):
    # case : dictionary
    if obj.__class__ == dict :
        try:
            if obj_path:
                d = obj.copy()
                for k in obj_path:
                    d = d[k]
                res_json[key] = d
        except Exception as e:
            if not nullable:
                raise ScanlogException("Fatal : error occured for required key %s" % key)
            # else error occured but key is not required
    # case : simple object
    elif obj != 'N/A':
        res_json[key] = obj
    elif not nullable:        
        raise ScanlogException("Fatal : key %s is N/A but is required" % key)
    # else obj is 'N/A' ans is ignored

def try_update_value(
    data_model: LogData,
    attr_path: str,
    obj_or_dict,
    dict_path: list = None,
    nullable: bool = True
):
    try:
        # Extract value from nested dict if dict_path is provided
        value = obj_or_dict
        if dict_path:
            for key in dict_path:
                value = value[key]

        # Traverse the attr_path in data_model to get the final target
        parts = attr_path.split(".")
        target = data_model
        for attr in parts[:-1]:
            if not hasattr(target, attr):
                raise AttributeError(attr)
            target = getattr(target, attr)

        final_attr = parts[-1]
        if not hasattr(target, final_attr):
            raise AttributeError(final_attr)

        setattr(target, final_attr, value)

    except KeyError as e:
        if nullable:
            return
        raise ScanlogException(f"Missing key {e} in provided data and nullable=False for path {dict_path}")

    except AttributeError as e:
        raise AttributeError(f"Fatal: Invalid attribute path '{e}' in 'MODEL_OBJECT.{attr_path}'")

    except Exception as e:
        if not nullable:
            raise ValueError(f"Fatal: Failed to assign value to 'MODEL_OBJECT.{attr_path}': {e}")



def general_param_subsection(data_model: LogData, data_json, data, obdata):
    try:
        all_unique_theory = np.unique(data.metadata['methods'])
        if len(all_unique_theory) > 1:
            theo_array = np.array(data.metadata['methods'])
            _, theo_indices = np.unique(theo_array, return_index=True)
            theo_indices.sort()
            theo_array = theo_array[theo_indices]
        else:
            theo_array = all_unique_theory
    except:
        theo_array = 'N/A'
        
    if theo_array.__class__ != str :
        if len(theo_array) > 0:
            theo_array = theo_array.tolist() if (theo_array != 'N/A').any() else 'N/A'
        else:
            theo_array = 'N/A'
            
    if len(all_unique_theory) > 0:
        all_unique_theory = all_unique_theory.tolist() if (all_unique_theory != 'N/A').any() else 'N/A'
    else:
        all_unique_theory = 'N/A'
        
    methods = data.metadata.get('methods', ['N/A'])

    try_update_value(data_model, "comp_details.general.package", data.metadata, ['package'])
    if data_model.comp_details.general.package != None:
        data_model.comp_details.general.package = data_model.comp_details.general.package.lower()
    try_update_value(data_model, "comp_details.general.package_version", data.metadata, ['package_version'])
    try_update_value(data_model, "comp_details.general.all_unique_theory", all_unique_theory)
    
    if len(methods) > 0:
        try_update_value(data_model, "comp_details.general.last_theory", methods[-1])
        
    try_update_value(data_model, "comp_details.general.list_theory", theo_array)
    try_update_value(data_model, "comp_details.general.functional", data.metadata, ['functional'])
    try_update_value(data_model, "comp_details.general.basis_set_name", data.metadata, ['basis_set'])
    
    # basis set Pickle version
    try:
        basis_str = pickle.dumps(data.gbasis, protocol=0)
        basis_hash = hashlib.md5(basis_str).hexdigest()
        try_update_value(data_model, "comp_details.general.basis_set", basis_str.decode()) # "%s"  % basis_str[2:-1])
        try_update_value(data_model, "comp_details.general.basis_set_md5", basis_hash)
    except:
        pass
    
    try_update_value(data_model, "comp_details.general.basis_set_size", data_json, ['properties', 'orbitals', 'basis number'])
    try_update_value(data_model, "comp_details.general.ao_names", data_json, ['atoms', 'orbitals', 'names'])
    
    try:
        data_model.comp_details.general.is_closed_shell = repr(len(data.moenergies) == 1 
                                                            or np.allclose(*data.moenergies, atol=1e-6))
    except:
        pass
    
    # TODO : 'integration grid' not found in data by cclib (is it because of version differences or just not present in all formats?)
    try_update_value(data_model, "comp_details.general.integration_grid", data_json, ['properties', 'integration grid'])
    try_update_value(data_model, "comp_details.general.solvent", data.metadata, ['solvent'])
    try_update_value(data_model, "comp_details.general.solvent_reaction_field", data.metadata, ['scrf'])
    try_update_value(data_model, "comp_details.general.scf_targets", data_json, ['optimization', 'scf', 'targets'])
    try_update_value(data_model, "comp_details.general.core_electrons_per_atoms", data_json, ['atoms', 'core electrons'])


def geometry_param_subsection(data_model: LogData, data_json, data, obdata):
    try_update_value(data_model,"comp_details.geometry.geometric_targets", data_json, ['optimization', 'geometric targets'])


def freq_param_subsection(data_model: LogData, data_json, data, obdata):
    try_update_value(data_model, "comp_details.freq.temperature", data_json, ['properties', 'temperature'])
    try_update_value(data_model, "comp_details.freq.anharmonicity", data_json, ['vibrations', 'anharmonicity constants'])


def td_param_subsection(data_model: LogData, data_json, data, obdata):
    et_states = data_json.get('transitions', {}).get('electronic transitions', None)
    if et_states:
        data_model.comp_details.excited_states.nb_et_states = len(et_states)
    ## TODO:
    # data_model.comp_details.excited_states.TDA = 'N/A' # TODO: test Tamm Damcoff approx.
    # data_model.comp_details.excited_states.et_sym_constraints = 'N/A/'


def wavefunction_results_subsection(data_model: LogData, data_json, data, obdata):
    try_update_value(data_model, "results.wavefunction.homo_indexes", data_json, ['properties', 'orbitals', 'homos'])
    try_update_value(data_model, "results.wavefunction.MO_energies", data_json, ['properties', 'orbitals', 'energies'])
    try_update_value(data_model, "results.wavefunction.MO_sym", data_json, ['properties', 'orbitals', 'molecular orbital symmetry'])
    # MO_number, MO_energies, MO_sym, MO_coefs

    try:
        try_update_value(data_model, "results.wavefunction.MO_number", data_json, ['properties', 'orbitals', 'MO number'], 
                            nullable=False) # not nullable in this context, exception catched.
        # TODO: Pb with energies, if NaN -> -inf
        data.moenergies[-1][np.isnan(data.moenergies[-1])] = -np.inf
        w_cut = np.where(data.moenergies[-1] > 10.)
        b_cut = min(max(w_cut[0][0] if len(w_cut[0]) > 0 else 0,
                        data.homos.max() + 31),
                    len(data.moenergies[-1]))
        try_update_value(data_model, "results.wavefunction.MO_number_kept", int(b_cut))

        # prune energies and sym
        try_update_value(data_model, "results.wavefunction.MO_energies", [moen[:b_cut] for moen in data_json['properties']['orbitals']['energies']])
        
        # Problem if there is no MO_sym like in ORCA !
        # TODO : Check if correct logique (if there is no MO_sys, while do we use a MO_Sym value to calculate it ?)
        if data_json.get('properties', {}).get('orbitals', {}).get('molecular orbital symmetry'):
            try_update_value(data_model, "results.wavefunction.MO_sym", [mosym[:b_cut] for mosym in data_json['properties']['orbitals']['molecular orbital symmetry']])
        # if "MO_sym" in section : 
        #     try_update_value(data_model, "results.wavefunction.MO_sym", [mosym[:b_cut] for mosym in section["MO_sym"]])
        
        # prune mocoeffs (Sparse version removed)
        mo_coefs = []
        # take last mocoeffs  (-2 with alpha/beta or -1)
        nb_coef = -2 if len(data.moenergies) == 2 else -1
        for a in data.mocoeffs[nb_coef:]:
            mo_coefs.append(a.tolist())
            
        # data insertion into JSON
        data_model.results.wavefunction.MO_coefs = mo_coefs
    except Exception as e:
        # partial MO data (qc lvl2 takes the decision)
        pass
    
    try_update_value(data_model, "results.wavefunction.total_molecular_energy", data_json, ['properties', 'energy', 'total'])
    
    # eV to Hartree conversion
    try: 
        try_update_value(data_model, "results.wavefunction.total_molecular_energy", data_json['properties']['energy']['total'] / CstHartree2eV)
    except:
        ## TODO: pb with SP
        pass # SP ? failure ?
    
    try_update_value(data_model, "results.wavefunction.Mulliken_partial_charges", data_json, ['properties', 'partial charges', 'mulliken'])
    
    try: 
        data_model.results.wavefunction.moments = data.moments 
    except:
        pass # Dipolar moments not present depending on the verbosity of log. 
    
    try:
        data_model.results.wavefunction.SCF_values = data_json['optimization']['scf']['values'][-1][-1] 
    except:
        pass
    
    try_update_value(data_model, "results.wavefunction.virial_ratio", data_json, ['optimization', 'scf', 'virialratio'])
    
    # TODO: # data_model.results.wavefunction.Hirshfeld_partial_charges  = 'N/A' # see scanlog
    try:
        data_model.results.wavefunction.Hirshfeld_partial_charges = data.atomcharges["hirshfeld"].tolist()
    except:
        pass


def geom_results_subsection(data_model: LogData, data_json, data, obdata):
    try_update_value(data_model, "results.geometry.nuclear_repulsion_energy_from_xyz", nuclear_repulsion_energy(data))
    try_update_value(data_model, "results.geometry.OPT_DONE", data_json, ['optimization', 'done'])
    try_update_value(data_model, "results.geometry.elements_3D_coords", data_json, ['atoms', 'coords', '3d'])
    try_update_value(data_model, "results.geometry.geometric_values", data_json, ['optimization', 'geometric values'])


def freq_results_subsection(data_model: LogData, data_json, data, obdata):
    try_update_value(data_model, "results.freq.entropy", data_json, ['properties', 'entropy'])
    try:
        try_update_value(data_model, "results.freq.entropy", float("%.9f" % data_json['properties']['entropy']))
    except:
        pass
    
    try_update_value(data_model, "results.freq.enthalpy", data_json, ['properties', 'enthalpy'])
    try_update_value(data_model, "results.freq.free_energy", data_json, ['properties', 'energy', 'free energy'])
    try_update_value(data_model, "results.freq.zero_point_energy", data_json, ['properties', 'zero point energy'])
    try_update_value(data_model, "results.freq.electronic_thermal_energy", data_json, ['properties', 'electronic thermal energy'])
    try_update_value(data_model, "results.freq.vibrational_freq", data_json, ['vibrations', 'frequencies'])
    try_update_value(data_model, "results.freq.vibrational_int", data_json, ['vibrations', 'intensities', 'IR'])
    
    # here NWChem
    try:
        data_model.results.freq.polarizabilities = data.polarizabilities[0].tolist()
    except:
        pass
    
    try_update_value(data_model, "results.freq.vibrational_sym", data_json, ['vibrations', 'vibration symmetry'])
    try_update_value(data_model, "results.freq.vibration_disp", data_json, ['vibrations', 'displacement'])
    
    # here Gaussian
    try_update_value(data_model, "results.freq.vibrational_anharms", data_json, ['vibrations', 'anharmonicity constants'])
    try_update_value(data_model, "results.freq.vibrational_raman", data_json, ['vibrations', 'intensities', 'raman'])

def td_results_subsection(data_model: LogData, data_json, data, obdata):
    try_update_value(data_model, "results.excited_states.et_energies", data_json, ['transitions', 'electronic transitions'])
    try_update_value(data_model, "results.excited_states.et_oscs", data_json, ['transitions', 'oscillator strength'])
    try_update_value(data_model, "results.excited_states.et_sym",  data_json, ['transitions', 'symmetry'])
    try_update_value(data_model, "results.excited_states.et_transitions", data_json, ['transitions', 'one excited config'])
    
    # here NWChem
    try_update_value(data_model, "results.excited_states.et_rot", data_json, ['transitions', 'rotatory strength'])
    try_update_value(data_model, "results.excited_states.et_eldips", data_json, ['transitions', 'electic transition dipoles'])
    try_update_value(data_model, "results.excited_states.et_veldips", data_json, ['transitions', 'velocity-gauge electric transition dipoles'])
    try_update_value(data_model, "results.excited_states.et_magdips", data_json, ['transitions', 'magnetic transition dipoles'])


def molecule_section(data_model: LogData, data_json, data, obdata, verbose=False):
    if "isChiral" in dir(obdata.OBMol) and "GetValence" in obdata.atoms[0].OBAtom:
        # openbabel 2
        data_model.molecule.chirality = obdata.OBMol.IsChiral()
        data_model.molecule.atoms_valence = [at.OBAtom.GetValence() for at in obdata.atoms]
    else:
        # openbabel 3
        data_model.molecule.chirality = obdata.OBMol.HasChiralityPerceived()
        # GetTotalValence() with implicit H, else GetExplicitValence()
        data_model.molecule.atoms_valence = [at.OBAtom.GetTotalValence() for at in obdata.atoms]
    # Start OpenBabel (all are mandatory)
    
    try:
        data_model.molecule.inchi = obdata.write("inchi").strip() # remove trailing \n
        data_model.molecule.smi = obdata.write("smi").split()[0]
        data_model.molecule.can = obdata.write("can").split()[0]
        data_model.molecule.monoisotopic_mass = obdata.OBMol.GetExactMass() # in Dalton
        
        connectivity_atom_pairs = []
        connectivity_bond_orders = []
        for i, a1 in enumerate(obdata.atoms):
            for j, a2 in enumerate(obdata.atoms):
                b = a1.OBAtom.GetBond(a2.OBAtom)
                if b is not None:
                    connectivity_atom_pairs.append((i, j))
                    connectivity_bond_orders.append(b.GetBondOrder())
        data_model.molecule.connectivity.atom_pairs = connectivity_atom_pairs
        data_model.molecule.connectivity.bond_orders = connectivity_bond_orders
    except:
        if verbose:
            traceback.print_exc()            
        raise ScanlogException("Reading mandatory data failed (Openbabel)")
    # End OpenBabel
    
    try_update_value(data_model, "molecule.formula", data_json, ['formula'])
    # CRITICAL TODO: formula versus inchi formula
    try_update_value(data_model, "molecule.nb_atoms", data_json, ['properties', 'number of atoms'])
    try_update_value(data_model, "molecule.nb_heavy_atoms", data_json, ['atoms', 'elements', 'heavy atom count'])
    try_update_value(data_model, "molecule.charge", data_json, ['properties', 'charge'])
    try_update_value(data_model, "molecule.multiplicity", data_json, ['properties', 'multiplicity'])
    try_update_value(data_model, "molecule.atoms_Z", data_json, ['atoms', 'elements', 'number'])
    
    # try_update_value(data_model, "molecule.atoms_masses", data.atommasses.tolist())
    # try_update_value(data_model, "molecule.nuclear_spins", data.nuclearspins.tolist())
    # try_update_value(data_model, "molecule.atoms_Zeff", data.atomzeff.tolist())
    # try_update_value(data_model, "molecule.nuclear_QMom", data.nuclearqmom.tolist())
    # try_update_value(data_model, "molecule.nuclear_gfactors", data.nucleargfactors.tolist())
    try_update_value(data_model, "molecule.starting_geometry", data.atomcoords[0,:,:].tolist())
    ## TODO: pb with SP
    try_update_value(data_model, "molecule.starting_energy", data_json, ["optimization", "scf", "scf energies"]) # in eV
    try:
        # eV to Hartree conversion
        try_update_value(data_model, "molecule.starting_energy", data_json["optimization"]["scf"]["scf energies"][0] / CstHartree2eV)
    except ScanlogException:
        pass #  SP ?
    try_update_value(data_model, "molecule.starting_nuclear_repulsion", nuclear_repulsion_energy(data, 0))


def parameters_section(data_model: LogData, data_json, data, obdata):
    # subsection : General parameters
    general_param_subsection(data_model, data_json, data, obdata)
    # subsection : Geometry 
    geometry_param_subsection(data_model, data_json, data, obdata)
    # subsection : Thermochemistry and normal modes
    freq_param_subsection(data_model, data_json, data, obdata)
    # subsection :  Excited states
    td_param_subsection(data_model, data_json, data, obdata)


def results_section(data_model: LogData, data_json, data, obdata):
    # subsection : Wavefunction
    wavefunction_results_subsection(data_model, data_json, data, obdata)
    # subsection : Geometry
    geom_results_subsection(data_model, data_json, data, obdata)
    # subsection : Thermochemistry and normal modes
    freq_results_subsection(data_model, data_json, data, obdata)
    # subsection : Excited states
    td_results_subsection(data_model, data_json, data, obdata)


def metadata_section(logfile, data_model: LogData, data_json, data, obdata):
    data_model.metadata.parser_version = scanlog_version
    data_model.metadata.log_file = os.path.basename(logfile)
    data_model.metadata.discretizable = False
    if data_model.results.wavefunction.MO_coefs != None:
        data_model.metadata.discretizable = True


def full_report(logfile, data_json, data, obdata, verbose=False):
    data_model = LogData()
    
    # section : Molecule
    molecule_section(data_model, data_json, data, obdata, verbose=verbose)
    # section : Computational details
    parameters_section(data_model, data_json, data, obdata)
    # section : Results
    results_section(data_model, data_json, data, obdata)
    # section : Metadata
    metadata_section(logfile, data_model, data_json, data, obdata)
    
    return data_model


def logfile_to_obj(logfile, verbose=False):
    # reading with cclib
    data = cclib.parser.ccopen(logfile).parse()
    data_json = json.loads(data.writejson())
    # openbabel sur XYZ
    obdata = pybel.readstring("xyz", data.writexyz())
    # construct raw data object 
    return full_report(logfile, data_json, data, obdata, verbose=verbose)


"""Check if logfile is archivable and candidate for a new entry.
"""       
def quality_check_lvl2(data_model: LogData, solver, verbose=False):
    qual = "True"
    qual2 = "True"
    
    # if not "basis_set_md5" in res_json["comp_details"]["general"].keys():
    #     qual = "False"
    if data_model.results.wavefunction.total_molecular_energy == None:
        qual = "False"
    # if OPT then res_json["results"]["wavefunction"]["MO_coefs"] needed
    # if 'OPT' in res_json["comp_details"]["general"]["job_type"]:
    #     if "MO_coefs" in res_json["results"]["wavefunction"].keys():
    #         qual2 = "True"
            # # If only OPT then qual = False (not archivable) ???
            # if len(res_json["comp_details"]["general"]["job_type"]) == 1:
            #     qual = "False"
    # TODO: mandatory job_type, package & package_version for import policy
    
    data_model.metadata.archivable = qual
    data_model.metadata.archivable_for_new_entry = qual2
    
    if verbose:
        print(">>> START QC lvl2 <<<")
        print("File:", data_model.metadata.log_file)
        print("Job type:", data_model.comp_details.general.job_type)
        print("Archivable:", data_model.metadata.archivable)
        print("Archivable for new entry:", data_model.metadata.archivable_for_new_entry)
        print(">>> END QC lvl2 <<<\n")


def process_logfile(config: Config, logfile, logfile_idx, log_storage_path="", verbose=False):
    solver = config.common_solver
    
    log_files = split_logfile(
        logfile, solver, log_storage_path=log_storage_path, verbose=verbose
    )
    data_list = []
    
    for log in log_files:
        data_model: LogData = logfile_to_obj(log, verbose=verbose)
        data_model.comp_details.general.job_type = config.logfiles[logfile_idx].type   
        # job_type_guess(res_json) # jobtype logique replaced and metadata.discretizable moved over to metadata section
        quality_check_lvl2(data_model, solver, verbose=verbose)
        data_list.append(data_model)
        
    return (log_files, data_list)


def process_logfile_list(config: Config, log_storage_path="", verbose=False):    
    logfilelist = [str(l.path.resolve()) for l in config.logfiles]
    
    data_list = []
    log_files = []
    
    for logfile_idx, logfile in enumerate(logfilelist):
        l, d = process_logfile(
            config, logfile, logfile_idx, log_storage_path=log_storage_path, verbose=verbose
        )
        log_files += l
        data_list += d

    return (log_files, data_list)
