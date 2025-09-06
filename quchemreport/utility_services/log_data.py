from typing import Any, Dict, List, Tuple, Union

from pydantic import BaseModel
import numpy as np

class CustomModel(BaseModel):
    model_config = {
        "validate_assignment": True,
    }

class Metadata(CustomModel):
    parser_version: str = None
    log_file: str = None
    discretizable: bool = None
    archivable: bool = None
    archivable_for_new_entry: bool = None


class MoleculeConnectivity(CustomModel):
    atom_pairs: List[Tuple[int, int]] = None
    bond_orders: List[int] = None


class Molecule(CustomModel):
    chirality: bool = None
    atoms_valence: List[int] = None
    inchi: str = None
    smi: str = None
    can: str = None
    monoisotopic_mass: float = None
    connectivity: MoleculeConnectivity = MoleculeConnectivity()
    formula: str = None
    nb_atoms: int = None
    nb_heavy_atoms: int = None
    charge: int = None
    multiplicity: int = None
    atoms_Z: List[int] = None
    # atoms_masses: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    # nuclear_spins: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    # atoms_Zeff: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    # nuclear_QMom: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    # nuclear_gfactors: List[float] = None # TODO: TO CHECK (Value unused / Maybe later)
    starting_geometry: List[List[float]] = None
    starting_energy: float = None
    starting_nuclear_repulsion: float = None

class CompDetailsGeneral(CustomModel):
    package: str = None
    package_version: str = None
    all_unique_theory: List[str] = None
    last_theory: str = None
    list_theory: List[str] = None
    functional: str = None
    basis_set_name: str = None
    basis_set: str = None
    basis_set_md5: str = None
    basis_set_size: int = None
    ao_names: List[str] = None
    is_closed_shell: bool = None
    integration_grid: str = None
    solvent: str = None
    solvent_reaction_field: str = None
    scf_targets: List[List[float]] = None
    core_electrons_per_atoms: List[int] = None
    job_type: List[str] = None

class CompDetailsGeometry(CustomModel):
    geometric_targets: List[float] = None


class CompDetailsFreq(CustomModel):
    temperature: Union[float, List[float]] = None # TODO : TO CHECK (DATA NOT FOUND IN SCANLOG & NOT SURE FOR TYPE)
    anharmonicity: List[List[float]] = None # TODO : TO CHECK (NOT SURE FOR TYPE)


class CompDetailsExcitedStates(CustomModel):
    nb_et_states: int = None
    # TDA: str = None # TODO: Maybe Later
    # et_sym_constraints: str = None # TODO: Maybe Later


class CompDetails(CustomModel):
    general: CompDetailsGeneral = CompDetailsGeneral()
    geometry: CompDetailsGeometry = CompDetailsGeometry()
    freq: CompDetailsFreq = CompDetailsFreq()
    excited_states: CompDetailsExcitedStates = CompDetailsExcitedStates()


class ResultsWaveFunction(CustomModel):
    homo_indexes: List[int] = None
    MO_energies: List[List[float]] = None
    MO_sym: List[List[str]] = None
    MO_number: int = None
    MO_number_kept: int = None
    MO_coefs: List[List[List[float]]] = None
    total_molecular_energy: float = None
    Mulliken_partial_charges: List[float] = None
    virial_ratio: float = None
    moments: List[List[float]] = None # List[List[float]]
    SCF_values: List[float] = None
    Hirshfeld_partial_charges: List[float] = None # TODO : TO CHECK (Not sure for the type) + COMMENTED IN SCANLOG BUT USED IN CALC_ORB
    A: float = None # TODO : TO CHECK (Not sure for the type)
    I: float = None # TODO : TO CHECK (Not sure for the type)
    Khi: float = None # TODO : TO CHECK (Not sure for the type)
    Eta: float = None # TODO : TO CHECK (Not sure for the type)
    Omega: float = None # TODO : TO CHECK (Not sure for the type)
    DeltaN: float = None # TODO : TO CHECK (Not sure for the type)
    fplus_lambda_mulliken: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fminus_lambda_mulliken: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fdual_lambda_mulliken: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fplus_lambda_hirshfeld: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fminus_lambda_hirshfeld: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    fdual_lambda_hirshfeld: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    CM5_partial_charges: List[float] = None  # TODO : TO CHECK (Not found in scanlong & Not sure for the type)

class ResultsGeometry(CustomModel):
    nuclear_repulsion_energy_from_xyz: float = None
    OPT_DONE: bool = None
    elements_3D_coords: List[float] = None
    geometric_values: List[List[float]] = None


class ResultsFreq(CustomModel):
    entropy: float = None
    enthalpy: float = None # TODO : TO CHECK (Not sure for the type)
    free_energy: float = None # TODO : TO CHECK (Not sure for the type)
    zero_point_energy: float = None # TODO : TO CHECK (Not sure for the type)
    electronic_thermal_energy: float = None # TODO : TO CHECK (Not sure for the type)
    vibrational_freq: List[float] = None # TODO : TO CHECK (Not sure for the type)
    polarizabilities: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    vibrational_int: List[float] = None # TODO : TO CHECK (Not sure for the type)
    vibrational_sym: List[str] = None # TODO : TO CHECK (Not sure for the type)
    vibration_disp: List[List[List[float]]] = None # TODO : TO CHECK (Not sure for the type)
    vibrational_anharms: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    vibrational_raman: List[float] = None # TODO : TO CHECK (Not sure for the type)


class ResultsExcitedStates(CustomModel):
    et_energies: Union[List[float], Dict[str, Any]] = None # TODO : TO CHECK (Not sure for the type)
    et_oscs: List[float] = None # TODO : TO CHECK (Not sure for the type)
    et_sym : List[str] = None # TODO : TO CHECK (Not sure for the type)
    et_transitions: List[List[Tuple[Tuple[int, int], Tuple[int, int], float]]] = None # TODO : TO CHECK (Not sure for the type)
    et_rot: List[float] = None # TODO : TO CHECK (Not sure for the type)
    et_eldips: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    et_veldips: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    et_magdips: List[List[float]] = None # TODO : TO CHECK (Not sure for the type)
    Tozer_lambda: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    d_ct: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    q_ct: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    mu_ct: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    e_barycenter: List[Any] = None # TODO : TO CHECK (Not sure for the type)
    hole_barycenter: List[Any] = None # TODO : TO CHECK (Not sure for the type)


class Results(CustomModel):
    wavefunction: ResultsWaveFunction = ResultsWaveFunction()
    geometry: ResultsGeometry = ResultsGeometry()
    freq: ResultsFreq = ResultsFreq()
    excited_states: ResultsExcitedStates = ResultsExcitedStates()


class LogData(CustomModel):
    metadata: Metadata = Metadata()
    molecule: Molecule = Molecule()
    comp_details: CompDetails = CompDetails()
    results: Results = Results()