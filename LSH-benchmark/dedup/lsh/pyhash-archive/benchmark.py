import timeit
import pyhash
import numpy as np

rng = np.random.default_rng(0)
mp = np.uint64((1 << 61) - 1)
values = rng.integers(low=0, high=np.iinfo(np.uint64).max, size=128, dtype=np.uint64)
val_32 = values.astype(np.uint32)

def hash_pyints(x):
    hash = int((1<<61)-1)
    total = 0
    for i in x:
        total += int(i)
    return total % hash

x = [2**64-1]*256
nx = np.array(x, dtype=np.uint64)
print(pyhash.hash(nx))
print(sum(x) % ((1 << 61) - 1))

print(round(timeit.timeit("pyhash.hash(nx)", globals=globals(), number=100000),ndigits=5))
# print(round(timeit.timeit("sum(values) % mp", globals=globals(), number=100000),ndigits=5))
print(round(timeit.timeit("sum(x) % ((1 << 61) - 1)", globals=globals(), number=100000),ndigits=5))
# print(round(timeit.timeit("rust_pyhash.hash(val_32.astype(np.uint64))", globals=globals(), number=100000),ndigits=5))
# print(round(timeit.timeit("hash_pyints(values)", globals=globals(), number=100000), ndigits=5))

