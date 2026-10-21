"""
subgraph_benchmark.py

Benchmark for the induced subgraph isomorphism solver in subgraph.py.

Each case generates random undirected graphs from a fixed seed, so runs are
reproducible and every configuration is timed on the same instances. In a
"planted" case, H is an induced subgraph of G on randomly chosen vertices, so a
match always exists, induced or not. In a "random" case, H is an independent
random graph.

Configurations named "fc..." solve induced subgraph isomorphism, and those
named "mono..." solve non-induced subgraph isomorphism (monomorphism); see
subgraph.py. Every result is checked: a returned mapping must pass
verify_mapping for its version of the problem, and configurations solving the
same version must agree on whether a match exists. Search counters ("nodes"
and "checks") are reported alongside time, since time alone varies a lot
between instances.

Usage:

    python examples/subgraph_benchmark.py
    python examples/subgraph_benchmark.py --configs fc --trials 10 --timeout 30

The per-run timeout uses SIGALRM, so it works on Linux and macOS only.
"""

import argparse
import random
import signal
import statistics
import time

from subgraph import find_subgraph, verify_mapping


# Solver configurations: name -> keyword arguments for find_subgraph
CONFIGS = {
    "fc": {},
    "fc+nbr": {"refine": "neighbors"},
    "fc+full": {"refine": "full"},
    "mono": {"induced": False},
    "mono+nbr": {"induced": False, "refine": "neighbors"},
}

# Benchmark cases: (name, n, m, edge probability, planted)
CASES = [
    ("planted n=30 m=8 p=0.3", 30, 8, 0.3, True),
    ("planted n=60 m=12 p=0.3", 60, 12, 0.3, True),
    ("planted n=100 m=20 p=0.2", 100, 20, 0.2, True),
    ("planted n=150 m=25 p=0.1", 150, 25, 0.1, True),
    ("planted n=100 m=15 p=0.5", 100, 15, 0.5, True),
    ("random n=30 m=8 p=0.3", 30, 8, 0.3, False),
    ("random n=40 m=10 p=0.5", 40, 10, 0.5, False),
]


class Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise Timeout


def random_graph(n, p, rng):
    """
    Returns the adjacency matrix of an undirected random graph G(n, p).
    """
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                A[i][j] = A[j][i] = 1
    return A


def make_instances(n, m, p, planted, trials, rng):
    """
    Returns a list of (G, H) pairs for one benchmark case.
    """
    instances = []
    for _ in range(trials):
        G = random_graph(n, p, rng)
        if planted:
            vs = rng.sample(range(n), m)
            H = [[G[a][b] for b in vs] for a in vs]
        else:
            H = random_graph(m, p, rng)
        instances.append((G, H))
    return instances


def run_one(G, H, kwargs, timeout):
    """
    Solves one instance. Returns (found, seconds, stats), where found is
    True, False, or None on timeout.
    """
    stats = {}
    signal.alarm(timeout)
    start = time.perf_counter()
    try:
        mapping = find_subgraph(G, H, stats=stats, **kwargs)
    except Timeout:
        return None, time.perf_counter() - start, stats
    finally:
        signal.alarm(0)
    seconds = time.perf_counter() - start
    if mapping is not None:
        induced = kwargs.get("induced", True)
        assert verify_mapping(G, H, mapping, induced=induced), "invalid mapping returned"
    return mapping is not None, seconds, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help=f"comma-separated configurations to run (default: all of {', '.join(CONFIGS)})")
    parser.add_argument("--trials", type=int, default=5, help="instances per case (default: 5)")
    parser.add_argument("--timeout", type=int, default=10, help="seconds allowed per instance (default: 10)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()

    configs = args.configs.split(",")
    for name in configs:
        if name not in CONFIGS:
            parser.error(f"unknown configuration {name!r}; choose from {', '.join(CONFIGS)}")

    signal.signal(signal.SIGALRM, _raise_timeout)

    print(f"trials={args.trials} timeout={args.timeout}s seed={args.seed}")
    print(f"{'case':28s} {'config':12s} {'mean ms':>10s} {'median ms':>10s} {'mean nodes':>12s} "
          f"{'mean checks':>12s} {'found':>6s} {'timeouts':>9s}")

    for case_name, n, m, p, planted in CASES:
        # One seed per case, so each case's instances don't depend on the others
        rng = random.Random(f"{args.seed}:{case_name}")
        instances = make_instances(n, m, p, planted, args.trials, rng)

        # (instance index, induced) -> found, from the first config that finished it
        answers = {}
        for config in configs:
            induced = CONFIGS[config].get("induced", True)
            times, nodes, checks = [], [], []
            found_count = timeouts = 0
            for i, (G, H) in enumerate(instances):
                found, seconds, stats = run_one(G, H, CONFIGS[config], args.timeout)
                times.append(seconds)
                nodes.append(stats.get("nodes", 0))
                checks.append(stats.get("checks", 0))
                if found is None:
                    timeouts += 1
                    continue
                found_count += found
                key = (i, induced)
                if key in answers:
                    assert answers[key] == found, f"{case_name}: configs disagree on instance {i}"
                else:
                    answers[key] = found

            print(f"{case_name:28s} {config:12s} {statistics.mean(times) * 1e3:10.1f} "
                  f"{statistics.median(times) * 1e3:10.1f} {statistics.mean(nodes):12.0f} "
                  f"{statistics.mean(checks):12.0f} {found_count:6d} {timeouts:9d}", flush=True)

    print("\nTimes, nodes and checks for timed-out instances are counted up to the timeout.")


if __name__ == "__main__":
    main()
