import os

DEFAULT_CONFIG = {
    "start_collection": "run_20250902_110438",
    "end_collection": "run_20250902_130518",
    "mongodb_port": int(os.getenv("SERVER_MONGODB_PORT", 27017)),
    "mongodb_host": "localhost",
    "database_name": "optimization",
    "plot_width": 1000,
    "plot_height": 600,
    "histogram_bins": 50,
}


class Config:
    def __init__(self, **kwargs):
        self.config = {**DEFAULT_CONFIG, **kwargs}

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def update(self, **kwargs):
        self.config.update(kwargs)

    def get(self, key, default=None):
        return self.config.get(key, default)
