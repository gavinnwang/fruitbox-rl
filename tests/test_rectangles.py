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


def test_rectangles():
    board = np.random.randint(1, 10, size=(10, 17))

    print("Board:")
    print(board)
    print()

    fast = sorted(find_valid_rectangles(board))
    brute = brute_force_rectangles(board)

    print("Fast algorithm rectangles:", len(fast))
    print("Brute force rectangles:", len(brute))
    print()

    print("First few rectangles (fast):")
    print(fast[:10])
    print()

    print("First few rectangles (brute):")
    print(brute[:10])
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
