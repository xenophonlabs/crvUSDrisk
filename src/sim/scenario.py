class Scenario:
    __slots__ = (
        "name",
        "description",
        "T",  # Number of steps
        "dt",  # Length of each step in seconds
    )

    def __init__(self, metadata: dict):
        """
        Generate the scenario from the stress
        test scenario config file.
        """
        pass
