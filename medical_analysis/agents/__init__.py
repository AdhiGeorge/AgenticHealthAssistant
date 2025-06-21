"""Medical agents package."""

from .base import MedicalAgent
from .orchestrator import OrchestratorAgent
from .cardiology import CardiologyAgent
from .neurology import NeurologyAgent
from .pulmonology import PulmonologyAgent
from .endocrinology import EndocrinologyAgent
from .gastroenterology import GastroenterologyAgent
from .hematology import HematologyAgent
from .nephrology import NephrologyAgent
from .rheumatology import RheumatologyAgent
from .infectious_disease import InfectiousDiseaseAgent
from .oncology import OncologyAgent
from .dermatology import DermatologyAgent
from .ophthalmology import OphthalmologyAgent
from .orthopedics import OrthopedicsAgent
from .psychiatry import PsychiatryAgent
from .pediatrics import PediatricsAgent
from .geriatrics import GeriatricsAgent

__all__ = [
    'MedicalAgent', 'OrchestratorAgent',
    'CardiologyAgent', 'NeurologyAgent', 'PulmonologyAgent', 'EndocrinologyAgent',
    'GastroenterologyAgent', 'HematologyAgent', 'NephrologyAgent', 'RheumatologyAgent',
    'InfectiousDiseaseAgent', 'OncologyAgent', 'DermatologyAgent', 'OphthalmologyAgent',
    'OrthopedicsAgent', 'PsychiatryAgent', 'PediatricsAgent', 'GeriatricsAgent'
] 