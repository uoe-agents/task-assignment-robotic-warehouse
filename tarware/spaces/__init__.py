from .MultiAgentGlobalObservationSpace import MultiAgentGlobalObservationSpace
from .MultiAgentPartialObservationSpace import \
    MultiAgentPartialObservationSpace

observation_map = {
    'partial': MultiAgentPartialObservationSpace,
    'global': MultiAgentGlobalObservationSpace
}