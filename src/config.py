import yaml


class Config:
    """
    Class to load and store configuration from a yaml file.
    """

    def __init__(self, path="./config.yml"):
        with open(path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __repr__(self):
        return str(self.config)

    def keys(self):
        return self.config.keys()

    def values(self):
        return self.config.values()
