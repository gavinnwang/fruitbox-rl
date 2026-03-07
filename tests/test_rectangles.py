import time
import numpy as np

from fruitbox_rl.algorithms.rectangle_finder import find_valid_rectangles


def brute_force_rectangles(board, target=10):
    rows, cols = board.shape
    actions = []

    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    if board[r1:r2+1, c1:c2+1].sum() == target:
                        actions.append((r1, c1, r2, c2))

    return sorted(actions)


def benchmark(fn, board, target=10, trials=100):
    times = []
    result = None

    for _ in range(trials):
        start = time.perf_counter()
        result = fn(board, target)
        end = time.perf_counter()
        times.append(end - start)

    avg = sum(times) / len(times)
    return result, avg, min(times), max(times)


def test_rectangles():
    board = np.random.randint(1, 10, size=(10, 17), dtype=np.int32)

    print("Board:")
    print(board)
    print()

    fast, fast_avg, fast_min, fast_max = benchmark(
        find_valid_rectangles, board, trials=200
    )
    brute, brute_avg, brute_min, brute_max = benchmark(
        brute_force_rectangles, board, trials=200
    )

    fast = sorted(fast)
    brute = sorted(brute)

    print("Fast algorithm rectangles:", len(fast))
    print("Brute force rectangles:", len(brute))
    print()

    print("First few rectangles (fast):")
    print(fast[:10])
    print()

    print("First few rectangles (brute):")
    print(brute[:10])
    print()

    print("Timing over 200 trials:")
    print(
        f"Fast   avg: {fast_avg*1000:.3f} ms   "
        f"min: {fast_min*1000:.3f} ms   "
        f"max: {fast_max*1000:.3f} ms"
    )
    print(
        f"Brute  avg: {brute_avg*1000:.3f} ms   "
        f"min: {brute_min*1000:.3f} ms   "
        f"max: {brute_max*1000:.3f} ms"
    )
    print()

    if fast_avg > 0:
        print(f"Speedup: {brute_avg / fast_avg:.2f}x")
    print()

    if fast == brute:
        print("TEST PASSED ✓")
    else:
        print("TEST FAILED ✗")
        print("Differences:")
        print("Fast only:", set(fast) - set(brute))
        print("Brute only:", set(brute) - set(fast))


if __name__ == "__main__":
    test_rectangles()
