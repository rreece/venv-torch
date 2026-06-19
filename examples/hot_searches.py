#!/usr/bin/env python3
"""
Hot Searches

You're building a feature for a search engine that tracks "hot searches" — the most recently used queries, up to some fixed capacity.

Implement a SearchCache class with:
- __init__(self, maxsize) — initialize with a max capacity
- add_search(self, query) — add a search query to the cache. If the cache is full, evict the least recently used entry. If the query is already in the cache, it should be treated as
recently used (but the size shouldn't grow).
- get_hot_searches(self) — return the current set of cached queries

Here's the behavior you need to satisfy:

cache = SearchCache(3)
cache.add_search(1)
assert set(cache.get_hot_searches()) == {1}
cache.add_search(1)
assert set(cache.get_hot_searches()) == {1}
cache.add_search(2)
assert set(cache.get_hot_searches()) == {1, 2}
cache.add_search(3)
assert set(cache.get_hot_searches()) == {1, 2, 3}
cache.add_search(4)
assert set(cache.get_hot_searches()) == {2, 3, 4}
cache.add_search(5)
assert set(cache.get_hot_searches()) == {3, 4, 5}
cache.add_search(4)
assert set(cache.get_hot_searches()) == {3, 4, 5}  # 4 re-accessed, 3 is now LRU
cache.add_search(3)
assert set(cache.get_hot_searches()) == {3, 4, 5}  # 3 re-accessed, 4 is now LRU

Take a moment to think about your data structure choice before you start coding. What would you use, and why?
"""

from collections import deque
from collections import OrderedDict


class SearchCacheDeque():
    def __init__(self, maxsize):
        """
        Initialize with a max capacity
        """
        self.cache = deque(maxlen=maxsize)

    def add_search(self, query):
        """
        Add a search query to the cache.
        If the cache is full, evict the least recently used entry.
        If the query is already in the cache, it should be treated as
        recently used (but the size shouldn't grow).
        """
        if query in self.cache:
            self.cache.remove(query)

        self.cache.append(query)

    def get_hot_searches(self):
        """
        Return the current set of cached queries
        """
        return set(self.cache)


class SearchCache():
    def __init__(self, maxsize):
        """
        Initialize with a max capacity
        """
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def add_search(self, query):
        """
        Add a search query to the cache.
        If the cache is full, evict the least recently used entry.
        If the query is already in the cache, it should be treated as
        recently used (but the size shouldn't grow).
        """

        if query in self.cache:
            self.cache.move_to_end(query, last=True)
            self.cache[query] += 1
        else:
            self.cache[query] = 1

        if len(self.cache) > self.maxsize:
            _, _ = self.cache.popitem(last=False)

    def get_hot_searches(self):
        """
        Return the current set of cached queries
        """
        return set(self.cache)


def main():
#    cache = SearchCacheDeque(3)
    cache = SearchCache(3)
    cache.add_search(1)
    assert set(cache.get_hot_searches()) == {1}
    cache.add_search(1)
    assert set(cache.get_hot_searches()) == {1}
    cache.add_search(2)
    assert set(cache.get_hot_searches()) == {1, 2}
    cache.add_search(3)
    assert set(cache.get_hot_searches()) == {1, 2, 3}
    cache.add_search(4)
    assert set(cache.get_hot_searches()) == {2, 3, 4}
    cache.add_search(5)
    assert set(cache.get_hot_searches()) == {3, 4, 5}
    cache.add_search(4)
    assert set(cache.get_hot_searches()) == {3, 4, 5}  # 4 re-accessed, 3 is now LRU
    cache.add_search(3)
    assert set(cache.get_hot_searches()) == {3, 4, 5}  # 3 re-accessed, 4 is now LRU
    cache.add_search(6)
    assert set(cache.get_hot_searches()) == {3, 4, 6}  # 5 was untouched since insertion, so it's LRU and gets evicted
    print("Pass!")


if __name__ == "__main__":
    main()
