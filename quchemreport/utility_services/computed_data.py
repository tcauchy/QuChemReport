from dataclasses import dataclass, fields, is_dataclass
from typing import List, Optional, Any, Tuple, Union, get_origin, get_args
import inspect

@dataclass
class Metadata:
    log_files: List[str]
    ref_log_file_idx: int
    output_dir: str


@dataclass
class MoleculeConnectivity:
    atom_pairs: List[Optional[List[Tuple[int, int]]]]
    # bond_orders: List[Optional[List[int]]]


@dataclass
class Molecule:
    charges: List[Optional[int]]
    multiplicity: List[Optional[int]]
    atoms_Z: List[Optional[List[int]]]
    monoisotopic_mass: Optional[float]
    inchi: Optional[str]
    smi: Optional[str]
    connectivity: MoleculeConnectivity
    formula: Optional[str]


@dataclass
class GeneralCompDetails:
    job_type: List[Optional[List[str]]]
    package: Optional[str]
    package_version: List[Optional[str]]
    last_theory: List[Optional[str]]
    functional: List[Optional[str]]
    basis_set_name: List[Optional[str]]
    basis_set_size: List[Optional[str]]
    is_closed_shell: List[Optional[bool]]
    integration_grid: List[Optional[str]]
    solvent: List[Optional[str]]
    scf_targets: List[Optional[List[List[float]]]]


@dataclass
class GeometryCompDetails:
    geometric_targets: List[Optional[List[float]]]


@dataclass
class FrequencyCompDetails:
    temperature: List[Optional[float]]
    anharmonicity: List[Optional[List[List[float]]]]


@dataclass
class ExcitedStatesCompDetails:
    nb_et_states: List[Optional[int]]


@dataclass
class ComputationDetails:
    general: GeneralCompDetails
    geometry: GeometryCompDetails
    freq: FrequencyCompDetails
    excited_states: ExcitedStatesCompDetails


@dataclass
class GeometryResults:
    elements_3D_coords: List[Optional[List[float]]]
    geometric_values: List[Optional[List[List[float]]]]
    nuclear_repulsion_energy_from_xyz: List[Optional[float]]


@dataclass
class FrequencyResults:
    vibrational_int: List[Optional[List[float]]]
    vibrational_freq: List[Optional[List[float]]]
    vibrational_sym: List[Optional[List[str]]]
    zero_point_energy: List[Optional[float]]
    electronic_thermal_energy: List[Optional[float]]
    enthalpy: List[Optional[float]]
    free_energy: List[Optional[float]]
    entropy: List[Optional[float]]


@dataclass
class ExcitedStatesResults:
    et_sym: List[Optional[List[str]]]
    et_energies: List[Optional[List[float]]]
    et_oscs: List[Optional[List[float]]]
    et_rot: List[Optional[List[Any]]]
    et_transitions: List[Optional[List[Any]]]
    et_eldips: List[Optional[List[List[float]]]]
    et_veldips: List[Optional[List[List[float]]]]
    et_magdips: List[Optional[List[List[float]]]]


@dataclass
class WavefunctionResults:
    total_molecular_energy: List[Optional[float]]
    homo_indexes: List[Optional[List[int]]]
    MO_energies: List[Optional[List[List[float]]]]
    Mulliken_partial_charges: Optional[List[float]]
    Hirshfeld_partial_charges: Optional[List[float]]
    CM5_partial_charges: Optional[List[float]]
    moments: List[Optional[List[List[float]]]]
    A: Optional[float]
    I: Optional[float]
    Khi: Optional[float]
    Eta: Optional[float]
    Omega: Optional[float]
    DeltaN: Optional[float]
    fplus_lambda_hirshfeld: Optional[List[float]]
    fminus_lambda_hirshfeld: Optional[List[float]]
    fdual_lambda_hirshfeld: Optional[List[float]]
    fplus_lambda_mulliken: Optional[List[float]]
    fminus_lambda_mulliken: Optional[List[float]]
    fdual_lambda_mulliken: Optional[List[float]]


@dataclass
class Results:
    geometry: GeometryResults
    freq: FrequencyResults
    excited_states: ExcitedStatesResults
    wavefunction: WavefunctionResults


@dataclass
class MoleculeDerivedData:
    topology_groups: List[Optional[int]]


@dataclass
class MODiagramsComputedData:
    spin: str
    labels: List[str]
    discretized_MO_voxels: Any
    grid_X: Any
    grid_Y: Any
    grid_Z: Any


@dataclass
class MODiagramsDerivedData:
    alpha: Optional[MODiagramsComputedData]
    beta: Optional[MODiagramsComputedData]
    restricted: Optional[MODiagramsComputedData]


@dataclass
class MOAnalysisDerivedData:
    MO_labels: List[Optional[List[List[str]]]]
    MO_indexes: List[Optional[List[List[int]]]]
    MO_diagrams: MODiagramsDerivedData


@dataclass
class MEPMapsDerivedData:
    rho_voxels: Optional[Any]
    MEP_voxels: Optional[Any]
    grid_X: Optional[Any]
    grid_Y: Optional[Any]
    grid_Z: Optional[Any]


@dataclass
class PopAnalysisDerivedData:
    Mulliken_mean: Optional[float]
    Mulliken_std: Optional[float]
    Hirshfeld_mean: Optional[float]
    Hirshfeld_std: Optional[float]
    CM5_partial_mean: Optional[float]
    CM5_partial_std: Optional[float]
    indices: Optional[List[int]]


@dataclass
class WavefunctionDerivedData:
    MO_analysis: MOAnalysisDerivedData
    MEP_maps: MEPMapsDerivedData
    pop_analysis: PopAnalysisDerivedData


@dataclass
class FukuiData:
    rho_voxels: Any
    grid_X: Any
    grid_Y: Any
    grid_Z: Any


@dataclass
class FukuiFunctionsDerivedData:
    SP_plus: Optional[FukuiData]
    SP_minus: Optional[FukuiData]
    dual: Optional[FukuiData]


@dataclass
class ReactivityDerivedData:
    fukui_functions: FukuiFunctionsDerivedData


@dataclass
class ExcitedStatesDerivedData:
    Tozer_lambda: List[Optional[List[float]]]
    d_ct: List[Optional[List[float]]]
    q_ct: List[Optional[List[float]]]
    TD_output: List[Optional[Any]]
    grid_X: List[Optional[Any]]
    grid_Y: List[Optional[Any]]
    grid_Z: List[Optional[Any]]
    all_data_dip: List[Optional[List[Any]]]
    abso_spectrum: List[Optional[Any]]
    CD_spectrum: List[Optional[Any]]
    xvalues: List[Optional[Any]]


@dataclass
class OptimizedExcitedStatesDerivedData:
    Tozer_lambda: List[Optional[List[float]]]
    d_ct: List[Optional[List[float]]]
    q_ct: List[Optional[List[float]]]
    TD_output: List[Optional[Any]]
    grid_X: List[Optional[Any]]
    grid_Y: List[Optional[Any]]
    grid_Z: List[Optional[Any]]
    all_data_dip: List[Optional[List[Any]]]
    emi_spectrum: List[Optional[Any]]
    CPL_spectrum: List[Optional[Any]]
    xvalues: List[Optional[Any]]

@dataclass
class DerivedData:
    molecule: MoleculeDerivedData
    wavefunction: WavefunctionDerivedData
    reactivity: ReactivityDerivedData
    excited_states: ExcitedStatesDerivedData
    optimized_excited_states: OptimizedExcitedStatesDerivedData


@dataclass
class ComputedData:
    metadata: Metadata
    molecule: Molecule
    comp_details: ComputationDetails
    results: Results
    derived: DerivedData


def initialize_field(field_type, n):
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle list of values for each logfile
    # List[T]
    if origin is list and args:
        elem_type = args[0]
        return [None for _ in range(n)]

    # Handle unique values
    # Optional[T] (Union[T, None])
    if origin is Union and type(None) in args:
        return None

    # Default case : return None
    return None


def init_dataclass(cls, n):
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    init_kwargs = {}
    for field in fields(cls):
        field_type = field.type

        # Special Case : Init metadata.log_files to "N/A"
        if cls.__name__ == "Metadata" and field.name == "log_files":
            init_kwargs[field.name] = ["N/A"] * n
            continue
        
        # Special Case : Init Metadata.ref_log_file_idx to 0
        if cls.__name__ == "Metadata" and field.name == "ref_log_file_idx":
            init_kwargs[field.name] = 0
            continue

        if inspect.isclass(field_type) and is_dataclass(field_type):
            init_kwargs[field.name] = init_dataclass(field_type, n)
        else:
            init_kwargs[field.name] = initialize_field(field_type, n)

    return cls(**init_kwargs)


def init_computed_data(n):
    if (n < 1):
        raise ValueError(f"The value n (number of log files) to initialize ComputedData must be at least 1 and not {n}.")
    
    return init_dataclass(ComputedData, n)