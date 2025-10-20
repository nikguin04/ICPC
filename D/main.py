import sys
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop
import math
from bisect import bisect_left, bisect_right

input = sys.stdin.readline


def read_ints(): return list(map(int, input().split()))
def read_str(): return input().strip()
def read_strs(): return input().strip().split()

N = int(read_str())
print(N)
Q = []
for i in range(0,N):
    i = read_ints()
    # X, Y, CLOSEST, CUR_DIST
    Q.append( (i[0], i[1], None, float("inf") ) )

def pytagoras(x0,y0,x1,y1):
    return math.sqrt( (x1-x0)**2 + (y1-y0)**2 )

for i in range(0,N):
    Q_checking = Q[i]
    for j in range(i+1,N):
        Q_comp = Q[j]
        #print(i, " ", j)
        dist = pytagoras(Q_checking[0], Q_checking[1], Q_comp[0], Q_comp[1])
        if dist < Q_checking[3]:
            Q_checking[2] = Q_comp
            Q_checking[3] = dist

start_cand = None
for i in range (0,N):
    
        
        
        


print(Q)