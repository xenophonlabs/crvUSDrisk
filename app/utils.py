"""Provides utility functions for the application."""
from copy import deepcopy


def load_markdown_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        out = file.read()
        out = out.replace("# ", "### ")
        out = out.replace("crvUSD Risk Assumptions and Limitations", "")
        return out


def clean_metadata(metadata):
    """Clean metadata for display."""
    metadata = deepcopy(metadata)
    metadata_ = metadata["template"].llamma.metadata
    metadata["bands_x"] = metadata_["llamma_params"]["bands_x"].copy()
    del metadata_["llamma_params"]["bands_x"]
    metadata["bands_y"] = metadata_["llamma_params"]["bands_y"].copy()
    del metadata_["llamma_params"]["bands_y"]
    for spool in metadata_["stableswap_pools_params"]:
        spool["coins"] = [c.symbol for c in spool["coins"]]
    return metadata
