import json
import logging
from dataclasses import dataclass

@dataclass
class Scenario:

    def __init__(self, fn: str):
        """
        Generate the scenario from the stress
        test scenario config file.
        """

        with open(fn, "r") as f:
            logging.info(f"Reading price config from {fn}.")
            config = json.load(f)

        self.name = config["name"]
        self.description = config["description"]
        self.N = config["N"]
        self.price_config = config["price_config"]
