import pathlib
import json
import datasets
from openai import OpenAI

client = OpenAI()

# Prompt example
example = """
# Question
Arthur owns a ski resort on a mountain. There are $n$ landing spots on the mountain numbered from $1$ to $n$ from the top to the foot of the mountain. The spots are connected with one-directional ski tracks. All tracks go towards the foot of the mountain, so there are no directed cycles formed by the tracks. There are at most two tracks leaving each spot, but many tracks may enter the same spot.

A skier can start skiing from one spot and stop in another spot if there is a sequence of tracks that lead from the starting spot and end in the ending spot. Unfortunately, recently there were many accidents, because the structure of the resort allows a skier to go through dangerous paths, by reaching high speed and endangering himself and the other customers. Here, a path is called dangerous, if it consists of at least two tracks.

Arthur wants to secure his customers by closing some of the spots in a way that there are no dangerous paths in the resort. When a spot is closed, all tracks entering and leaving that spot become unusable. 

Formally, after closing some of the spots, there should not be a path that consists of two or more tracks.

Arthur doesn't want to close too many spots. He will be happy to find any way to close at most $\frac{4}{7}n$ spots so that the remaining part is safe. Help him find any suitable way to do so.


-----Input-----

The first line contains a single positive integer $T$ — the number of test cases. $T$ test case description follows.

The first line of each description contains two integers $n$ and $m$ ($1 \leq n \leq 2 \cdot 10^5$) — the number of landing spots and tracks respectively.

The following $m$ lines describe the tracks. Each of these lines contains two integers $x$ and $y$ ($1 \leq x < y \leq n$) — indices of the starting and finishing spots for the respective track. It is guaranteed that at most two tracks start at each spot. There may be tracks in which starting and finishing spots both coincide.

It is guaranteed that the sum of $n$ over all test cases does not exceed $2 \cdot 10^5$.


-----Output-----

For each test case, print a single integer $k$ ($0 \leq k \leq \frac{4}{7}n$) — the number of spots to be closed. In the next line, print $k$ distinct integers — indices of all spots to be closed, in any order.

If there are several answers, you may output any of them. Note that you don't have to minimize $k$. It can be shown that a suitable answer always exists.


-----Example-----
Input
2
4 6
1 2
1 3
2 3
2 4
3 4
3 4
7 6
1 2
1 3
2 4
2 5
3 6
3 7

Output
2
3 4 
4
4 5 6 7 



-----Note-----

In the first sample case, closing any two spots is suitable.

In the second sample case, closing only the spot $1$ is also suitable.

# Code
```
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
```

# Plan
We can solve this problem by framing it as removing all paths from a DAG that have length greater than or equal to 3.

1. **Read Input:**
   - Read the number of test cases, `T`.
   - For each test case, read the number of landing spots (`n`) and the number of tracks (`m`).

2. **Initialize Data Structures:**
   - Create an adjacency list to store the outgoing connections for each spot.
   - Create an in-degree count array to track the number of incoming connections to each spot.

3. **Process Tracks:**
   - For each track, read the start (`a`) and end (`b`) spots.
   - Adjust the spots to zero-indexed values.
   - Update the adjacency list and increment the in-degree for the destination spot `b`.

4. **Topological Sorting:**
   - Use Kahn's algorithm for topological sorting to assign levels to nodes/landing spots.
   - Initialize a queue with all nodes that have zero in-degree and process each node by visiting its neighbors, updating their in-degrees, and enqueuing them if their in-degree becomes zero.

5. **Determine Nodes to Remove:**
   - Use the levels derived from the topological sort to identify any nodes that would result in paths of length 3 or more:
     - If a node has a level difference of 3 or more with its descendants, mark it for removal.
   - Ensure the total number of nodes removed does not exceed the allowed bound (e.g., at most \( \frac{4}{7}n \)).

6. **Output Results:**
   - For each test case, print the number of nodes to be removed and their indices.

End plan.
"""

# create plans for 10 examples
N = 10

dataset = datasets.load_dataset("codeparrot/apps", split="train")

for x in dataset:
    prompt = f"""You are an expert coder. Given a question and coding solution, explain the plan behind the solution.

Here is an example.
{example}

Please complete the following.
# Question
{x["question"]}

# Code
{x["solutions"][0]}

# Plan"""
    chat_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    import pdb; pdb.set_trace()
    chat_completion.choices[0]

# store plans in
output_file = pathlib.Path("examples/apps/train/gpt-plans.json")
