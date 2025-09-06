from dataclasses import dataclass
from typing import List, Union
from quchemreport.utility_services.log_data import LogData

@dataclass
class ValidatedData:
    job_types: List[List[str]]
    nres_noES: List[float]
    charges: List[int]
    charge_ref: int
    discret_proc: bool
    mo_viz_done: bool
    ref_log_file_idx: int
    data_for_discretization: LogData