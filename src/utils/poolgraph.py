"""
Module providing an undirected graph data structure to represent
liquidity pools and their connections. The graph implements a 
DFS algorithm to find all cycles of a given length.
"""
from typing import List, Tuple, Set, Dict
from ..trades import Swap, Cycle
from ..modules import ExternalMarket
from ..types import SimPoolType
from ..logging import get_logger


logger = get_logger(__name__)


def shared_address(
    p1: SimPoolType, p2: SimPoolType, used: Set[str] | None = None
) -> Set[str]:
    """Check if two pools share coins by checking their addrs."""
    assert p1 != p2, ValueError("Cannot share coins with self.")

    c1 = [c.lower() for c in p1.coin_addresses]
    c2 = [c.lower() for c in p2.coin_addresses]

    shared = set(c1) & set(c2)
    if used:
        shared -= used
    assert len(shared) <= 1, NotImplementedError(
        f"We assume at most one shared coin. {type(p1), type(p2)}"
    )
    return shared


def get_shared_idxs(p1: SimPoolType, p2: SimPoolType) -> Tuple[int, int]:
    """Get the index of the shared coin in each pool."""
    shared = shared_address(p1, p2).pop()

    # FIXME inefficient to recreate c1 and c2
    c1 = [c.lower() for c in p1.coin_addresses]
    c2 = [c.lower() for c in p2.coin_addresses]

    return c1.index(shared), c2.index(shared)


class PoolGraph:
    """Undirected graph of liquidity pools."""

    def __init__(self, pools: List[SimPoolType]):
        self.pools = pools
        self.graph = self.create_graph()

    def create_graph(self) -> Dict[SimPoolType, List[SimPoolType]]:
        """Create an undirected graph of liquidity pools."""
        graph: Dict[SimPoolType, List[SimPoolType]] = {}
        for pool in self.pools:
            assert len(pool.coin_addresses) == 2, NotImplementedError(
                "Only 2-coin pools"
            )
            graph[pool] = []
            for other in self.pools:
                if other != pool and self.can_traverse(pool, other, set()):
                    graph[pool].append(other)
        return graph

    def find_cycles(self, n: int = 3) -> List[Cycle]:
        """
        Find all valid cycles of length `n` in the graph.
        Cycles are:
        1. Connected - each pool is connected to the next pool in the cycle
        by a shared coin.
        2. Unique - each coin is only used once in the cycle.
        """
        # TODO currently assumes only one shared coin between
        # any two pools.
        assert len(self.pools) >= n, ValueError("Not enough pools to form a cycle.")
        cycles: List[Cycle] = []
        for pool in self.pools:
            self.dfs(pool, [pool], set(), cycles, n)
        valid = self.validate(cycles)
        logger.info("Found %d valid cycles of length %d.", len(valid), n)
        return valid

    def can_traverse(self, curr: SimPoolType, nxt: SimPoolType, used: Set[str]) -> bool:
        """
        Determine if we can traverse from `curr` to `nxt` in the graph. This
        outputs `True` if the two pools share a coin that has not been used
        previously in the cycle.

        Parameters
        ----------
        curr : SimPoolType
            Current pool in the cycle.
        nxt : SimPoolType
            Next pool in the cycle.
        used : Set[str]
            Set of coin addresses that have already been used in the cycle.

        Returns
        -------
        bool
            `True` if we can traverse from `curr` to `nxt` in the graph.

        Note
        ----
        We don't traverse between External Markets.
        """
        if isinstance(curr, ExternalMarket) and isinstance(nxt, ExternalMarket):
            # Don't traverse between external markets
            return False
        return bool(shared_address(curr, nxt, used))

    def update_used_coins(self, used: Set[str], curr: SimPoolType, nxt: SimPoolType):
        """Update the set of used coins in the cycle."""
        used.update(shared_address(curr, nxt))

    def revert_used_coins(self, used: Set[str], curr: SimPoolType, nxt: SimPoolType):
        """Revert the set of used coins in the cycle."""
        used.difference_update(shared_address(curr, nxt))

    def construct_cycle(self, path: List[SimPoolType], n: int) -> Cycle:
        """Construct a Cycle object."""
        trades = []
        for i, pool in enumerate(path):
            nxt = path[(i + 1) % n]
            idx, _ = get_shared_idxs(pool, nxt)  # token out
            trades.append(Swap(pool, idx ^ 1, idx, 0))
        return Cycle(trades)

    # pylint: disable=too-many-arguments
    def dfs(
        self,
        curr: SimPoolType,
        path: List[SimPoolType],
        used: Set[str],
        cycles: List[Cycle],
        n: int,
    ):
        """
        Perform a depth-first search to find cycles of length `n`.

        Parameters
        ----------
        curr : SimPoolType
            Current pool in the cycle.
        path : List[SimPoolType]
            List of pools in the cycle.
        used : Set[str]
            Set of coin addresses that have already been used in the cycle.
        cycles : List[Cycle]
            List of cycles found so far.
        n : int
            Length of the cycle.
        """
        if len(path) == n:
            # Ensure cycle is closed
            shared = shared_address(path[0], path[-1], used)
            if bool(shared):
                cycles.append(self.construct_cycle(path, n))
            return

        for nxt in self.graph[curr]:
            if nxt in path:
                # Only visit each pool once per cycle
                continue
            if self.can_traverse(curr, nxt, used):
                path.append(nxt)
                self.update_used_coins(used, curr, nxt)
                self.dfs(nxt, path, used, cycles, n)
                # Backtrack
                path.pop()
                self.revert_used_coins(used, curr, nxt)

    def validate(self, cycles: List[Cycle]) -> List[Cycle]:
        """
        Filter for cycles that only have one ExternalMarket,
        and it's at the end of the cycle.

        TODO consider other kinds of cycles.
        """
        valid = []
        for cycle in cycles:
            pools = []
            for trade in cycle.trades:
                if not isinstance(trade, Swap):
                    raise NotImplementedError("Only Swap trades supported.")
                pools.append(trade.pool)
            if not isinstance(pools[-1], ExternalMarket):
                continue
            for pool in pools[:-1]:
                if isinstance(pool, ExternalMarket):
                    continue
            valid.append(cycle)
        return valid

    def test_cycles(self, cycles: List[Cycle], n: int):
        """
        Test that the cycles are valid:
        1. Cycle correctness (incl. closure)
        2. Cycle length
        3. Unique coin usage
        TODO move this to a unit test file
        """
        for cycle in cycles:
            pools = []
            for trade in cycle.trades:
                if not isinstance(trade, Swap):
                    raise NotImplementedError("Only Swap trades supported.")
                pools.append(trade.pool)
            assert cycle.n == n, "Wrong length"
            used = set()
            for i in range(n):
                used.update(shared_address(pools[i], pools[(i + 1) % n]))
                assert len(used) == i + 1
