from enum import Enum


class LoadType(Enum):
    PV = "PV"
    EV = "EV"
    HP = "HP"
    DHW = "DHW"
    TOTAL = "TOTAL"
