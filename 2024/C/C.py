N = int(input())

out = []
if N % 2 != 0:
    N -= 3
    out += [3]

out += ["2" for _ in range(N//2)]

print(len(out))
print(" ".join(map(str, out)))
