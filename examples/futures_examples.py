#!/usr/bin/env python3
"""
Examples of using concurrent.futures.
"""


import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def slow_square(n):
    time.sleep(0.5)
    return n * n


# --- Example 1: executor.map ---
# Like built-in map(), but runs in parallel.
# Results come back in submission order.

def run_map():
    inputs = range(8)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(slow_square, inputs))
    print(f"map:          {results}  ({time.perf_counter() - t0:.1f}s)")


# --- Example 2: executor.submit + Future ---
# submit() returns a Future immediately.
# .result() blocks until that future is done.

def run_submit():
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(slow_square, n) for n in range(8)]
        results = [f.result() for f in futures]
    print(f"submit:       {results}  ({time.perf_counter() - t0:.1f}s)")


# --- Example 3: as_completed ---
# Process results as they finish, not in submission order.
# Useful when tasks have variable runtimes and you want to act on each result ASAP.

def run_as_completed():
    inputs = [3, 1, 4, 1, 5, 9, 2, 6]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(slow_square, n): n for n in inputs}
        for f in as_completed(futures):
            n = futures[f]
            print(f"  {n}^2 = {f.result()}")
    print(f"as_completed: done in {time.perf_counter() - t0:.1f}s")


# --- Example 4: ProcessPoolExecutor ---
# Drop-in replacement for ThreadPoolExecutor.
# Use for CPU-bound work to bypass the GIL.

def run_processes():
    inputs = range(8)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(slow_square, inputs))
    print(f"processes:    {results}  ({time.perf_counter() - t0:.1f}s)")


def main():
    print("=== map ===")
    run_map()

    print("\n=== submit ===")
    run_submit()

    print("\n=== as_completed ===")
    run_as_completed()

    print("\n=== ProcessPoolExecutor ===")
    run_processes()

    print("\nDone.")


if __name__ == "__main__":
    main()
