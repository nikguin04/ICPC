N, K = map(int, input().split())
deck1 = list(map(int, input().split()))
deck2 = list(map(int, input().split()))

n1, n2 = len(deck1), len(deck2)
dp = [0]*(n2+1)
for i in range(n1):
    prev = 0
    for j in range(1, n2+1):
        cur = dp[j]
        if deck1[i] == deck2[j-1]:
            dp[j] = prev + 1
        elif dp[j-1] > dp[j]:
            dp[j] = dp[j-1]
        prev = cur

print(dp[n2])
