import sys
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop
from bisect import bisect_left, bisect_right

input = sys.stdin.readline


def read_ints(): return list(map(int, input().split()))
def read_str(): return input().strip()
def read_strs(): return input().strip().split()


def main():
    N, E = read_ints()
    ms = 0
    prev_E = 0
    if N == 1: 
        print("infinity")
        return
    
    while True:
        if E > 0:
            ms += E
        else:
            ms += 1
            
        # if (ms % N == 0 and ms % (E+1)):
        #     print("infinity")
        #     return   
            
            
        if (ms % N == 0):
            E += 1
        else: 
            E -= 1
        
        # if (E + prev_E == N):
        #     print("infinity")
        #     return        

        if E == 0:
            print(ms)
            return 
        
        prev_E = E
        
    print("stopped")
        
            
        
            
    
            
            
main()