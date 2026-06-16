#!/usr/bin/env python3
"""
Example of using threading
"""


import queue
import threading
import time


def run_broken():
    counter = 0

    def increment():
        nonlocal counter
        for _ in range(100_000):
            counter += 1  # read-add-write, can be interrupted between steps

    threads = [threading.Thread(target=increment) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    print(f"broken:  {counter} (expected 400000)")


def run_fixed():
    counter = 0
    lock = threading.Lock()

    def increment():
        nonlocal counter
        for _ in range(100_000):
            with lock:
                counter += 1

    threads = [threading.Thread(target=increment) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    print(f"fixed:   {counter} (expected 400000)")


def run_producer_consumer():
    q = queue.Queue(maxsize=5)  # buffer holds at most 5 items

    def producer(n_items):
        for i in range(n_items):
            q.put(i)            # blocks if queue is full
            print(f"produced {i}  (qsize={q.qsize()})")
            time.sleep(0.05)
        q.put(None)             # sentinel: signals consumer to stop

    def consumer():
        while True:
            item = q.get()      # blocks if queue is empty
            if item is None:
                break
            print(f"  consumed {item}")
            q.task_done()
            time.sleep(0.1)     # consumer is slower than producer

    p = threading.Thread(target=producer, args=(10,))
    c = threading.Thread(target=consumer)
    p.start()
    c.start()
    p.join()
    c.join()


def main():
    run_broken()
    run_fixed()
    print("")
    run_producer_consumer()
    print("Done.")


if __name__ == "__main__":
    main()
