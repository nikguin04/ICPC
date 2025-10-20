import sys

input = sys.stdin.readline

N,M,K = map(int, input().split())

# kids = [-1]*N # number: {group}
groups = {k:set() for k in range(N)}

for i in range(M):
    a, b = map(int, input().split())
    a,b = a-1,b-1

    groups[a].add(b)
    groups[b].add(a)

    groups[a].update(groups[b])
    groups[b].update(groups[a])

    # if groups[a] != kids[b]:
    #     gb = groups.pop(kids[b])
    #     for b_kid in gb:
    #         kids[b_kid] = kids[a] # put b kid in group a
    #     groups[kids[a]].update(gb) # put all kids in group b in group a

print(groups)

# Now we have groups, check length
#print(created_kids, " ", N)

for g in groups.values():
    if K > len(g):
        print("impossible")
        exit(0)

group_game_index = {g: 0 for g in groups}

# for i in range(0,N):
#     g = kids[i]
#     idx = group_game_index[g]
#     group_game_index[g] = (group_game_index[g]+1)%K
#     print(idx+1, end=" ")

