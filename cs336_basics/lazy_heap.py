import heapq


class LazyHeap:
    def __init__(self, pair_counts: dict[tuple[int, int], int]):
        self.pair_counts = pair_counts
        self.deleted: dict[tuple[int, tuple[int, int]], int] = {}
        self.heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(self.heap)

    def __getitem__(self, pair: tuple[int, int]) -> int:
        return self.pair_counts.get(pair, 0)

    def __setitem__(self, pair: tuple[int, int], count: int):
        #  it needs delete old pair to create new pair
        if pair in self.pair_counts:
            # if count is the same, do nothing
            if self.pair_counts[pair] == count:
                return
            item = (-self.pair_counts[pair], pair)
            self.deleted[item] = self.deleted.get(item, 0) + 1

        # if count <= 0, pair will be deleted
        if count <= 0:
            if pair in self.pair_counts:
                del self.pair_counts[pair]
            return

        self.pair_counts[pair] = count
        heapq.heappush(self.heap, (-count, pair))

    def __len__(self) -> int:
        return len(self.pair_counts)

    def __repr__(self) -> str:
        return f"LazyHeap(pair_counts={self.pair_counts}, heap={self.heap}), deleted={self.deleted}"

    def pop(self) -> tuple[tuple[int, int], int]:
        """Pop the most frequent pair from the heap"""
        while self.heap:
            item = heapq.heappop(self.heap)
            if item in self.deleted:
                self._cleanup_deleted(item)
                continue

            pair, count = item[1], -item[0]
            # delete pair
            del self.pair_counts[pair]

            return pair, count

    def top(self) -> tuple[tuple[int, int], int]:
        """Get the most frequent pair from the heap"""
        while self.heap:
            item = self.heap[0]
            if item in self.deleted:
                heapq.heappop(self.heap)
                self._cleanup_deleted(item)
                continue
            return item[1], -item[0]

    def _cleanup_deleted(self, item: tuple[int, tuple[int, int]]):
        if self.deleted[item] == 1:
            del self.deleted[item]
        else:
            self.deleted[item] -= 1
