from pydantic import BaseModel, Field, field_validator, model_validator, validator, root_validator
from typing import List, Optional, Union, Dict
from pathlib import Path
import warnings
import os

SUPPORTED_SOLVERS = {
    "adf", "dalton", "firefly", "gamess", "gamess-uk", "gaussian",
    "jaguar", "molcas", "molpro", "mopac", "nbo", "nwchem",
    "orca", "psi4", "q-chem", "turbomole"
}

ALLOWED_CALCULATION_TYPES = {"sp", "opt", "opt_es", "freq", "freq_es", "td", "nmr"}

class LogFile(BaseModel):
    path: Path
    type: List[str] # Single string accepted but converted into a list
    reference: Optional[bool] = False
    excited_state_number: Optional[int] = None

    @field_validator("path")
    def check_file_exists(cls, v: Path) -> Path:
        if not v.is_file():
            raise FileNotFoundError(f"Log file does not exist: {v}")
        return v

    @field_validator("type", mode="before")
    def validate_type_values(cls, v) -> List[str]:
        if isinstance(v, str):
            v = [v]
        elif not isinstance(v, list):
            raise TypeError(f"type must be a str or list of str, got {type(v)}")
        
        if not v:
            raise ValueError("type must not be empty")

        invalid_types = [t for t in v if t not in ALLOWED_CALCULATION_TYPES]
        if invalid_types:
            raise ValueError(f"type contains invalid values: {invalid_types}. Allowed: {ALLOWED_CALCULATION_TYPES}")
        return v

    @model_validator(mode="after")
    def check_excited_state_number(self) -> "LogFile":
        types = [self.type] if isinstance(self.type, str) else self.type or []
        has_es_type = any("_es" in t for t in types)

        if has_es_type:
            if self.excited_state_number is None:
                raise ValueError("excited_state_number must be set when a type contains '_es'")
            if self.excited_state_number <= 0:
                raise ValueError("excited_state_number must be > 0")
        else:
            if self.excited_state_number is not None:
                raise ValueError("excited_state_number should only be set if type contains '_es'")
        return self

class QualityChecks(BaseModel):
    formula: bool = True
    theory: bool = True
    nuclear_repulsion: bool = True
    charge: bool = True
    multiplicity: bool = True
    ground_state_optimization: bool = True

class QualityControl(BaseModel):
    method_consistency: str
    checks: QualityChecks

    @field_validator("method_consistency")
    def method_consistency_must_be_valid(cls, v: str) -> str:
        allowed = {"strict", "lax"}
        if v not in allowed:
            raise ValueError(f"method_consistency must be 'strict' or 'lax'. Not '{v}' .")
        return v


class MoleculeInclude(BaseModel):
    molecule_representation: bool
    molecule_identification: List[str]
    atomic_coordinates: bool

    @field_validator("molecule_identification", mode="before")
    def check_identification_values(cls, v):
        allowed = {"directory name", "formula", "charge", "spin multiplicity", "monoisotopic mass", "inchi"}
        if not isinstance(v, list):
            raise TypeError("molecule_identification must be a list")
        for item in v:
            item_lower = item.lower()
            if item_lower not in allowed:
                raise ValueError(f"Invalid molecule_identification value: {item_lower}. Must be one of {allowed}")
        return v


class WavefunctionMOAnalysis(BaseModel):
    MO_list: List[str]
    energies: bool
    MO_diagrams: bool
    
    @model_validator(mode="after")
    def check_mo_list_if_needed(self) -> "WavefunctionMOAnalysis":
        if (self.energies or self.MO_diagrams) and (not self.MO_list or len(self.MO_list) == 0):
            raise ValueError("MO_list must not be empty when energies or MO_diagrams is True")
        return self


class WavefunctionInclude(BaseModel):
    MO_analysis: WavefunctionMOAnalysis
    MEP_maps: bool
    population_analysis: bool


class FrequenciesInclude(BaseModel):
    thermo_data: bool
    IR_spectrum: bool


class ReactivityInclude(BaseModel):
    CDFT_indices: bool
    fukui_functions: bool


class ExcitedStatesSelection(BaseModel):
    mode: str
    first_n: Optional[int] = None

    @field_validator("mode")
    def mode_must_be_valid(cls, v: str) -> str:
        allowed = {"dominant_only", "first_n", "all_states"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}. Not '{v}' .")
        return v

    @model_validator(mode="after")
    def check_first_n_if_needed(self) -> "ExcitedStatesSelection":
        if self.mode == "first_n":
            if self.first_n is None:
                raise ValueError("first_n must be set when mode is 'first_n'")
            if self.first_n <= 0:
                raise ValueError("first_n must be a positive integer")
        return self


class ExcitedStatesInclude(BaseModel):
    selection: ExcitedStatesSelection
    transitions: bool
    absorption_spectrum: bool
    CD_spectrum: bool
    electron_density_difference: bool
    
    @model_validator(mode="after")
    def selection_required_if_transitions_or_edd(cls, values):
        selection = values.selection
        if (values.transitions or values.electron_density_difference) and not selection:
            raise ValueError(
                "A selection mode must be provided when transitions or "
                "electron_density_difference is enabled."
            )
        return values


class OptimizedExcitedStatesInclude(BaseModel):
    transitions: bool
    emission_spectrum: bool
    CPL_spectrum: bool
    electron_density_difference: bool


class NMRShifts(BaseModel):
    enabled: bool
    elements: List[str]


class NMRCouplings(BaseModel):
    enabled: bool
    elements: List[str]


class NMRInclude(BaseModel):
    NMR_shifts: NMRShifts
    NMR_couplings: NMRCouplings

class OutputInclude(BaseModel):
    molecule: MoleculeInclude
    computational_details: bool
    wavefunction: WavefunctionInclude
    frequencies: FrequenciesInclude
    reactivity: ReactivityInclude
    excited_states: ExcitedStatesInclude
    optimized_excited_states: OptimizedExcitedStatesInclude
    # TODO : TO IMPLEMENT : NMR output is yet to be implemented
    # NMR: NMRInclude

class Output(BaseModel):
    output_path: str
    molecule_designated_name: str
    format: str # html | markdown | pdf # TODO : Output formats are subject to changes
    verbosity: str # full | si | text
    include: OutputInclude

    @field_validator("format")
    def validate_format(cls, v: str) -> str:
        allowed = {"html", "markdown", "pdf"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Output format must be one of {allowed}. Not '{v_lower}' .")
        return v_lower

    @field_validator("verbosity")
    def validate_verbosity(cls, v: str) -> str:
        allowed = {"full", "si", "text"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Output verbosity must be one of {allowed}. Not '{v_lower}' .")
        return v_lower


class CameraView(BaseModel):
    mode: str = "auto"  # auto | preset | manual
    preset: Optional[str] = None  # x, y, z
    view_file: Optional[Path] = None

    @field_validator("mode")
    def mode_must_be_valid(cls, v: str) -> str:
        allowed = {"auto", "preset", "manual"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Camera view mode must be one of: {allowed}. Not '{v_lower}' .")
        return v_lower

    @model_validator(mode="after")
    def check_mode_dependencies(self) -> "CameraView":
        if self.mode == "preset":
            allowed_preset = {"x", "y", "z"}
            if self.preset not in allowed_preset:
                raise ValueError(f"When mode is 'preset', 'preset' must be one of: {allowed_preset}. Not '{self.preset}' .")
        elif self.mode == "manual":
            if self.view_file is None:
                raise ValueError(f"When mode is 'manual', 'view_file' must be provided. Not '{self.view_file}' .")
        return self


class LoggingSettings(BaseModel):
    console: bool
    level: str

    @field_validator("level")
    def level_must_be_valid(cls, v: str) -> str:
        allowed = {"debug", "info", "warning", "error"}
        if v not in allowed:
            raise ValueError(f"Logging level must be one of: {allowed}. Not '{v}' .")
        return v

class Config(BaseModel):
    common_solver: str
    logfiles: List[LogFile]
    quality_control: QualityControl
    output: Output
    logging: LoggingSettings
    # camera_view: CameraView

    @field_validator("common_solver", mode="before")
    def normalize_and_validate_solver(cls, v: str) -> str:
        if not v:
            raise ValueError("common_solver must be specified")
        v_lower = v.lower()
        if v_lower not in SUPPORTED_SOLVERS:
            raise ValueError(f"common_solver '{v}' is not supported. Supported solvers: {SUPPORTED_SOLVERS}")
        return v_lower

    @field_validator("logfiles")
    def logfiles_must_not_be_empty(cls, v: List[LogFile]) -> List[LogFile]:
        if not v:
            raise ValueError("At least one logfile must be specified")
        return v