def process_test_cases():
    num_cases = int(input())
    cases = []
    for n in range(num_cases):
        n, k = map(int, input().split())
        barrels = list(map(int, input().split()))
        cases.append((n, k, barrels))
    return num_cases, cases


def max_possible_difference(n, k, barrels):
    sorted_barrels = list(sorted(barrels, reverse=True))
    return sum(sorted_barrels[:k+1])


def main():
    num_test_cases, test_cases = process_test_cases()
    for test_case in test_cases:
        print(max_possible_difference(*test_case))

main()
