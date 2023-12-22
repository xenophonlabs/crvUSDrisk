import cProfile
import pstats
import pdb
from datetime import datetime
from src.sim import run_scenario
from src.logging import get_logger

logger = get_logger(__name__)


def with_analysis(num_iter, ncpu):
    with cProfile.Profile() as pr:
        try:
            run_scenario("baseline", "wstETH", num_iter=num_iter, ncpu=ncpu)
        except Exception:
            pdb.post_mortem()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    stats.dump_stats("./logs/sim.prof")


def without_analysis(num_iter, ncpu):
    start = datetime.now()
    logger.info("Start: %s", start)
    run_scenario("baseline", "wstETH", num_iter=num_iter, ncpu=ncpu)
    end = datetime.now()
    logger.info("End: %s", end)
    diff = end - start
    logger.info("Diff: %s", diff)


if __name__ == "__main__":
    num_iter = 100
    ncpu = 12
    with_analysis(num_iter, ncpu)
    # without_analysis(num_iter, ncpu)
