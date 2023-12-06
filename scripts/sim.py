import logging
import cProfile
import pstats
import pdb
from src.sim import sim

logging.basicConfig(
    filename="./logs/sim.log", level=logging.INFO, format="%(asctime)s %(message)s"
)

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        try:
            sim("baseline")
        except Exception:
            pdb.post_mortem()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats("./logs/sim.prof")
