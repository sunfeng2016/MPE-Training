REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC
from .task_controller import TaskMAC
from .maic_controller import MAICMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["task_mac"] = TaskMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC
REGISTRY['maic_mac'] = MAICMAC