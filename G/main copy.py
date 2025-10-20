import sys

input = sys.stdin.readline

N,M,K = map(int, input().split())

kids = {} # number: {group}
groups = {}

for i in range(M):
    a, b = map(int, input().split())

    if a not in kids:
        groups[a] = set([a])
        kids[a] = a
    
    if b not in kids:
        groups[b] = set([b])
        kids[b] = b
        

    if kids[a] != kids[b]:
        gb = groups.pop(kids[b])
        for b_kid in gb:
            kids[b_kid] = kids[a] # put b kid in group a
        groups[kids[a]].update(gb) # put all kids in group b in group a

# Now we have groups, check length
if N != len(kids): # Not all kids have a friend group
    print("impossible")
    exit(0)

for g in groups.values():
    if K > len(g):
        print("impossible")
        exit(0)

group_game_index = {g: 0 for g in groups}

for i in range(1,N+1):
    g = kids[i]
    idx = group_game_index[g]
    group_game_index[g] = (group_game_index[g]+1)%K
    print(idx+1, end=" ")

