from .fql_agent import FQLAgent
from .sacbc_agent import SACBCAgent
from .qsm_agent import QSMAgent
from .dsrl_agent import DSRLAgent
from .ifql_agent import IFQLAgent
from .pqsmpd_agent import PQSMPDAgent

agents = {
    "fql": FQLAgent,
    "sacbc": SACBCAgent,
    "qsm": QSMAgent,
    "pqsmpd": PQSMPDAgent,
    "dsrl": DSRLAgent,
    "ifql": IFQLAgent,
}
