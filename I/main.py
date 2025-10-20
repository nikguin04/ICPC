import sys

input = sys.stdin.readline

N, M = map(int, input().split())

score = [0] * N
followers = [set() for _ in range(0, N)]

for i in range(M):
    person, follow = map(int, input().split())
    person-=1; follow-=1
    if follow in followers[person]:
        score[person] -= 1
    else:
        score[follow] += 1
        
    followers[follow].add(person)

biggest = 0
for i in range(1,N):
    if score[i] > score[biggest]:
        biggest = i
print(biggest + 1, " ", score[biggest])