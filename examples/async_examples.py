#!/usr/bin/env python3
"""
Examples of using asyncio
"""


import asyncio
import time


# --- Example 1: basic async/await ---

async def fetch(name, delay):
    print(f"start  {name}")
    await asyncio.sleep(delay)   # yields control back to event loop
    print(f"finish {name}")
    return f"{name} result"


async def run_sequential():
    t0 = time.perf_counter()
    a = await fetch("A", 1.0)
    b = await fetch("B", 1.0)
    print(f"sequential: {time.perf_counter() - t0:.1f}s  →  {a}, {b}")


async def run_concurrent():
    t0 = time.perf_counter()
    a, b = await asyncio.gather(
        fetch("A", 1.0),
        fetch("B", 1.0),
    )
    print(f"concurrent: {time.perf_counter() - t0:.1f}s  →  {a}, {b}")


# --- Example 2: producer/consumer with asyncio.Queue ---

async def producer(q, n_items):
    for i in range(n_items):
        await q.put(i)
        print(f"produced {i}  (qsize={q.qsize()})")
        await asyncio.sleep(0.05)
    await q.put(None)  # sentinel


async def consumer(q):
    while True:
        item = await q.get()
        if item is None:
            break
        print(f"  consumed {item}")
        await asyncio.sleep(0.1)


async def run_producer_consumer():
    q = asyncio.Queue(maxsize=5)
    await asyncio.gather(
        producer(q, 10),
        consumer(q),
    )


# --- main ---

async def main():
    print("=== sequential ===")
    await run_sequential()

    print("\n=== concurrent (gather) ===")
    await run_concurrent()

    print("\n=== producer/consumer ===")
    await run_producer_consumer()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
