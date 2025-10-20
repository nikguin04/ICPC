import sys
import math

input = sys.stdin.readline

S = input().strip().split()
N = int(S[0])

inputs = [input().strip() for _ in range(N)]

curr = int(inputs[0])
limit = math.ceil((curr+0.1)/10)*10
speeds = [curr]

for i in range(1, N):
    sign = inputs[i]
    curr = 0
    if (sign == "/"):
        curr = limit
    else:
        curr = int(sign)
        if (curr >= limit):
            limit = math.ceil((curr+0.1)/10)*10
    speeds.append(curr)
    
print("\n".join(map(str, speeds)))