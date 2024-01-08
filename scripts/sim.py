import cProfile
import pstats
import pdb
import argparse
import pickle
from datetime import datetime
from multiprocessing import cpu_count
from src.sim import run_scenario
from src.logging import get_logger

logger = get_logger(__name__)


def with_analysis(scenario, markets, num_iter, ncpu):
    with cProfile.Profile() as pr:
        try:
            global output
            output = run_scenario(scenario, markets, num_iter=num_iter, ncpu=ncpu)
        except Exception:
            pdb.post_mortem()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    stats.dump_stats("./logs/sim.prof")
    return output


def without_analysis(scenario, markets, num_iter, ncpu):
    start = datetime.now()
    global output
    output = run_scenario(scenario, markets, num_iter=num_iter, ncpu=ncpu)
    end = datetime.now()
    diff = end - start
    logger.info("Total runtime: %s", diff)
    return output


def analysis_help():
    print("Call `output.summary` for a DF of summary metrics.")
    print("Call `output.plot_runs(<metric_id>)` to plot the input metric for all runs.")
    print("Call `output.metric_map` to get a list of metrics and their ids.")
    print("Call `output.plot_summary()` to plot histograms of summary metrics.")
    print("Call `output.data[i].plot_prices()` to plot the prices used for run `i`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate price config for sim.")
    parser.add_argument(
        "scenario", type=str, help="Scenario to simulate (from src/configs/scenarios)."
    )
    parser.add_argument("markets", type=str, help="Comma-separated list of markets.")
    parser.add_argument("num_iter", type=int, help="Number of runs.", default=10)
    parser.add_argument(
        "-mp",
        "--multiprocess",
        action="store_true",
        help="Multiprocess?",
        required=False,
    )
    args = parser.parse_args()

    markets = args.markets.split(",")[0]  # TODO handle multiple markets

    if args.multiprocess:
        ncpu = cpu_count()
    else:
        ncpu = 1

    output = with_analysis(args.scenario, args.markets, args.num_iter, ncpu)
    # output = without_analysis(args.scenario, args.markets, args.num_iter, ncpu)
    logger.info("Done. Call `analysis_help()` for more info in interactive mode.")

    with open(f"results/{args.scenario}.pkl", "wb") as f:
        pickle.dump(output, f)
