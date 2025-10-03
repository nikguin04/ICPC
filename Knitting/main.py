import sys

N, P = map(int, sys.stdin.readline().split())

print(0 if N % P == 0 else (N - P) % (2 * P))