import sys
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop
from bisect import bisect_left, bisect_right

input = sys.stdin.readline


def read_ints(): return list(map(int, input().split()))
def read_str(): return input().strip()
def read_strs(): return input().strip().split()
