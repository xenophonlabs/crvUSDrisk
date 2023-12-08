import logging
import cProfile
import pstats
import pdb
from src.sim import sim
from src.logging import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        try:
            sim("baseline")
        except Exception:
            pdb.post_mortem()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    stats.dump_stats("./logs/sim.prof")
