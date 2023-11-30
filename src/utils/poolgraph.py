import logging
from typing import List, Tuple, Set, Dict, Optional
from ..typing import SimPoolType
from ..trades import Swap, Cycle
from ..modules import ExternalMarket

# TODO add proper pool typing instead of Any


def shared_address(
    p1: SimPoolType, p2: SimPoolType, used: Set[Optional[str]] = set()
) -> Set[str]:
    """Check if two pools share coins by checking their addrs."""
    assert p1 != p2, ValueError("Cannot share coins with self.")

    c1 = [c.lower() for c in p1.coin_addresses]
    c2 = [c.lower() for c in p2.coin_addresses]

    shared = set(c1) & set(c2) - used
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
    def __init__(self, pools: List[SimPoolType]):
        self.pools = pools
        self.graph = self.create_graph()

    def create_graph(self) -> Dict[SimPoolType, List[SimPoolType]]:
        graph: Dict[SimPoolType, List[SimPoolType]] = {}
        for pool in self.pools:
            assert len(pool.coin_addresses) == 2, NotImplementedError(
                "Only 2-coin pools"
            )
            graph[pool] = []
            for other in self.pools:
                if other != pool and bool(shared_address(pool, other)):
                    graph[pool].append(other)
        return graph

    def find_cycles(self, n: int = 3) -> List[Cycle]:
        # TODO currently assumes only one shared coin between
        # any two pools.
        assert len(self.pools) >= n, ValueError("Not enough pools to form a cycle.")
        cycles: List[Cycle] = []
        for pool in self.pools:
            self.dfs(pool, [pool], set(), cycles, n)
        valid = self.validate(cycles)
        logging.info(f"Found {len(valid)} valid cycles of length {n}.")
        self.test_cycles(valid, n)  # TODO remove
        return valid

    def can_traverse(
        self, curr: SimPoolType, nxt: SimPoolType, used: Set[Optional[str]]
    ) -> bool:
        if isinstance(curr, ExternalMarket) and isinstance(nxt, ExternalMarket):
            # Don't traverse between external markets
            return False
        return bool(shared_address(curr, nxt, used))

    def update_used_coins(
        self, used: Set[Optional[str]], curr: SimPoolType, nxt: SimPoolType
    ):
        used.update(shared_address(curr, nxt))

    def revert_used_coins(
        self, used: Set[Optional[str]], curr: SimPoolType, nxt: SimPoolType
    ):
        used.difference_update(shared_address(curr, nxt))

    def construct_cycle(self, path: List[SimPoolType], n: int) -> Cycle:
        trades = []
        for i, pool in enumerate(path):
            nxt = path[(i + 1) % n]
            idx, _ = get_shared_idxs(pool, nxt)  # token out
            trades.append(Swap(pool, idx ^ 1, idx, None))
        return Cycle(trades)

    def dfs(
        self,
        curr: SimPoolType,
        path: List[SimPoolType],
        used: Set[Optional[str]],
        cycles: List[Cycle],
        n: int,
    ):
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
        Filter for cycles that only has one ExternalMarket,
        and it's at the end of the cycle.

        TODO does this make sense @Vishesh?
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
        # Testing
        # 1. Cycle correctness (incl. closure)
        # 2. Cycle length
        # 3. Unique coin usage
        # TODO move this to a unit test file
        for cycle in cycles:
            pools = []
            for trade in cycle.trades:
                if not isinstance(trade, Swap):
                    raise NotImplementedError("Only Swap trades supported.")
                pools.append(trade.pool)
            assert cycle.n == n, "Wrong length"
            used = shared_address(pools[0], pools[1])
            assert len(used) == 1
            used.update(shared_address(pools[1], pools[2]))
            assert len(used) == 2
            used.update(shared_address(pools[2], pools[0]))
            assert len(used) == 3
