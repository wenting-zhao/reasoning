import sys
from collections import deque

input = sys.stdin.readline

def process_test_cases():
    T = int(input())
    test_cases = []
    for _ in range(T):
        n, m = map(int, input().split())
        tracks = []
        for __ in range(m):
            a, b = map(int, input().split())
            tracks.append((a, b))
        test_cases.append((n, m, tracks))
    return test_cases

def remove_nodes_to_avoid_long_paths(n, m, tracks):
    adjacency_list = [[] for _ in range(n)]
    in_degree = [0] * n

    for a, b in tracks:
        a -= 1  # Convert to zero-indexed
        b -= 1  # Convert to zero-indexed
        adjacency_list[a].append(b)
        in_degree[b] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    level = [0] * n
    while queue:
        node = queue.popleft()
        for neighbor in adjacency_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
            level[neighbor] = max(level[neighbor], level[node] + 1)

    removal_set = set()
    for node in range(n):
        for neighbor in adjacency_list[node]:
            for next_neighbor in adjacency_list[neighbor]:
                if level[next_neighbor] >= level[node] + 2:
                    removal_set.add(next_neighbor)

    if len(removal_set) > (4 * n) // 7:
        removal_set = set(list(removal_set)[:(4 * n) // 7])

    return list(removal_set)

def main():
    test_cases = process_test_cases()
    results = []

    for n, m, tracks in test_cases:
        nodes_to_remove = remove_nodes_to_avoid_long_paths(n, m, tracks)
        results.append((len(nodes_to_remove), nodes_to_remove))

    for node_count, nodes in results:
        print(node_count)
        if node_count > 0:
            print(" ".join(map(str, (node + 1 for node in nodes))))  # Convert back to one-indexed

if __name__ == "__main__":
    main()
