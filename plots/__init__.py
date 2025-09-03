from dataclasses import dataclass
from plots.mongo_client import MyMongoClient
from plots.objectives import MyObjectivePlotter


@dataclass
class Config:
    start_collection: str = "run_20250902_110438"
    end_collection: str = "run_20250902_130518"
    mongodb_port: int = 27017
    mongodb_host: str = "localhost"
    database_name: str = "optimization"
    plot_width: int = 1000
    plot_height: int = 600
    histogram_bins: int = 50


__version__ = "1.0.0"
__all__ = [
    "Config",
    "MyMongoClient",
    "MyObjectivePlotter",
]
