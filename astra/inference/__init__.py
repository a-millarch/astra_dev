from astra.inference.pipeline import InferenceSession, InferenceResult, SHAPResult
from astra.inference.patient_context import PatientContext
from astra.inference.data_prep import (
    prepare_single_patient,
    prepare_from_raw_ehr,
    prepare_patient_from_csv,
)
from astra.inference.simulation import SimulationRunner, SimulationResult, SimulationStep

# High-level API (external deployments start here — see docs/HANDOFF.md)
from astra.inference.api import (
    AstraPredictor,
    AstraPredictorError,
    PatientNotFoundError,
    TimestampBeforeAdmissionError,
    ArtifactError,
)
from astra.inference.responses import (
    TimeAxis,
    ProbabilityCurve,
    PredictionResponse,
    ExplanationResponse,
    DifferentialExplanationResponse,
)
from astra.inference.datasource import (
    PatientDataSource,
    CSVDataSource,
    InMemoryDataSource,
)
from astra.inference.patient_store import (
    set_data_source,
    get_data_source,
    clear_data_source,
)
