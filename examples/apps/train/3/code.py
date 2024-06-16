def parse_test_cases():
    num_cases = input()
    cases = []
    for n in num_cases:
        n, k = map(int, input().split())
        barrels = list(map(int, input().split()))
        cases.append((n, k, barrels))
    return cases


def max_possible_difference(n, k, barrels):
    sorted_barrels = list(sorted(barrels, reverse=True))
    return sum(sorted_barrels[:k])


def main():
    num_test_cases, test_cases = parse_test_cases
    for test_case in test_cases:
        print(max_possible_difference(*test_case))
