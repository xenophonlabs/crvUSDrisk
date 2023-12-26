import sys
import cProfile
import pstats
import pdb
from datetime import datetime
from multiprocessing import cpu_count
from src.sim import run_scenario
from src.logging import get_logger

logger = get_logger(__name__)


def with_analysis(num_iter, ncpu):
    with cProfile.Profile() as pr:
        try:
            global output
            output = run_scenario("baseline", "wstETH", num_iter=num_iter, ncpu=ncpu)
        except Exception:
            pdb.post_mortem()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    stats.dump_stats("./logs/sim.prof")


def without_analysis(num_iter, ncpu):
    start = datetime.now()
    global output
    output = run_scenario("baseline", "wstETH", num_iter=num_iter, ncpu=ncpu)
    end = datetime.now()
    diff = end - start
    logger.info("Total runtime: %s", diff)


def analysis_help():
    print("Call `output.summary` for a DF of summary metrics.")
    print("Call `output.plot_runs(<metric_id>)` to plot the input metric for all runs.")
    print("Call `output.metric_map` to get a list of metrics and their ids.")
    print("Call `output.plot_summary()` to plot histograms of summary metrics.")
    print("Call `output.data[i].plot_prices()` to plot the prices used for run `i`.")


if __name__ == "__main__":
    num_iter = 10
    # with_analysis(num_iter, cpu_count())
    without_analysis(num_iter, cpu_count())
    logger.info("Done. Call `analysis_help()` for more info in interactive mode.")
