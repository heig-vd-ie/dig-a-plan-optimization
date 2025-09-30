from enum import Enum


class LoadProfileType(Enum):
    DHW = "DHW"
    PV = "PV"
    EV = "EV"
    HP = "HP"
    TOTAL = "TOTAL"
