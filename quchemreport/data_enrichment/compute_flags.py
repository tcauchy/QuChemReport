from dataclasses import dataclass, fields
from quchemreport.config.config import Config


@dataclass
class ComputeFlags:
    compute_molecule_representations: bool = True
    compute_all_MO_labels_and_indexes: bool = True
    compute_MO_diagrams: bool = True
    compute_MEP_maps: bool = True
    compute_population_analysis_indices: bool = True
    compute_CDFT_global_indices: bool = True
    compute_fukui_functions: bool = True
    compute_es_Abso_spectrum: bool = True
    compute_es_CD_spectrum: bool = True
    compute_es_transitions_and_EDD: bool = True
    compute_optimized_es_emi_spectrum: bool = True
    compute_optimized_es_CPL_spectrum: bool = True
    compute_optimized_es_transitions_and_EDD: bool = True


# COMPUTE_FLAG_RULES links each ComputeFlags field to one or more attribute paths
# inside config.output.include. For each flag, if any of the specified config
# attributes are True, the flag is set to True; otherwise, it is False.
# This provides a clear and editable way to control which computation steps
# are activated based on user-selected outputs.
COMPUTE_FLAG_RULES = {
    "compute_molecule_representations": ["molecule.molecule_representation"],
    "compute_all_MO_labels_and_indexes": [
        "wavefunction.MO_analysis.energies",
        "wavefunction.MO_analysis.MO_diagrams"
    ],
    "compute_MO_diagrams": ["wavefunction.MO_analysis.MO_diagrams"],
    "compute_MEP_maps": ["wavefunction.MEP_maps"],
    "compute_population_analysis_indices": ["wavefunction.population_analysis"],
    "compute_CDFT_global_indices": ["reactivity.CDFT_indices"],
    "compute_fukui_functions": ["reactivity.fukui_functions"],
    "compute_es_Abso_spectrum": ["excited_states.absorption_spectrum"],
    "compute_es_CD_spectrum": ["excited_states.CD_spectrum"],
    "compute_es_transitions_and_EDD": [
        "excited_states.transitions",
        "excited_states.electron_density_difference"
    ],
    "compute_optimized_es_emi_spectrum": ["optimized_excited_states.emission_spectrum"],
    "compute_optimized_es_CPL_spectrum": ["optimized_excited_states.CPL_spectrum"],
    "compute_optimized_es_transitions_and_EDD": [
        "optimized_excited_states.transitions",
        "optimized_excited_states.electron_density_difference"
    ],
}


def get_value_from_path(root, path: str):
    current = root
    for part in path.split('.'):
        current = getattr(current, part)
    return current


def resolve_compute_flags(config: Config) -> ComputeFlags:
    include = config.output.include

    expected_fields = {f.name for f in fields(ComputeFlags)}
    rule_keys = set(COMPUTE_FLAG_RULES.keys())

    missing = expected_fields - rule_keys
    extra = rule_keys - expected_fields

    if missing:
        raise ValueError(f"Missing compute rules for: {missing}")
    if extra:
        raise ValueError(f"Unknown compute rules (not in ComputeFlags): {extra}")

    computed_values = {}

    for key, paths in COMPUTE_FLAG_RULES.items():
        if not isinstance(paths, list):
            paths = [paths]

        values = []
        for path in paths:
            try:
                val = get_value_from_path(include, path)
            except AttributeError as e:
                raise RuntimeError(f"Invalid path '{path}' for flag '{key}': {e}")
            values.append(val)

        computed_values[key] = any(values)

    return ComputeFlags(**computed_values)
