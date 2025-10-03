# Competitive Programming Algorithm Reference

## 1. Binary Search

**C++:**

```cpp
int binary_search(vector<int>& arr, int target) {
    int l = 0, r = arr.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (arr[m] == target) return m;
        if (arr[m] < target) l = m + 1;
        else r = m - 1;
    }
    return -1;
}
```

**Python:**

```python
def binary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        m = (l + r) // 2
        if arr[m] == target:
            return m
        if arr[m] < target:
            l = m + 1
        else:
            r = m - 1
    return -1
```

## 2. Depth-First Search (DFS)

**C++:**

```cpp
void dfs(int node, vector<vector<int>>& adj, vector<bool>& vis) {
    vis[node] = true;
    for (int neighbor : adj[node]) {
        if (!vis[neighbor]) {
            dfs(neighbor, adj, vis);
        }
    }
}
```

**Python:**

```python
def dfs(node, adj, vis):
    vis[node] = True
    for neighbor in adj[node]:
        if not vis[neighbor]:
            dfs(neighbor, adj, vis)
```

## 3. Breadth-First Search (BFS)

**C++:**

```cpp
void bfs(int start, vector<vector<int>>& adj, int n) {
    vector<bool> vis(n, false);
    queue<int> q;
    q.push(start);
    vis[start] = true;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        for (int neighbor : adj[node]) {
            if (!vis[neighbor]) {
                vis[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

**Python:**

```python
from collections import deque

def bfs(start, adj, n):
    vis = [False] * n
    q = deque([start])
    vis[start] = True
    while q:
        node = q.popleft()
        for neighbor in adj[node]:
            if not vis[neighbor]:
                vis[neighbor] = True
                q.append(neighbor)
```

## 4. Dijkstra's Shortest Path

**C++:**

```cpp
vector<int> dijkstra(int start, vector<vector<pair<int,int>>>& adj, int n) {
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    dist[start] = 0;
    pq.push({0, start});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

**Python:**

```python
import heapq

def dijkstra(start, adj, n):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

## 5. Union-Find (Disjoint Set Union)

**C++:**

```cpp
class DSU {
    vector<int> parent, rank;
public:
    DSU(int n) : parent(n), rank(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    bool unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        if (rank[x] < rank[y]) swap(x, y);
        parent[y] = x;
        if (rank[x] == rank[y]) rank[x]++;
        return true;
    }
};
```

**Python:**

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, x, y):
        x, y = self.find(x), self.find(y)
        if x == y:
            return False
        if self.rank[x] < self.rank[y]:
            x, y = y, x
        self.parent[y] = x
        if self.rank[x] == self.rank[y]:
            self.rank[x] += 1
        return True
```

## 6. Kruskal's MST

**C++:**

```cpp
struct Edge { int u, v, w; };

int kruskal(vector<Edge>& edges, int n) {
    sort(edges.begin(), edges.end(), [](auto& a, auto& b) { return a.w < b.w; });
    DSU dsu(n);
    int mst_cost = 0;
    for (auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            mst_cost += e.w;
        }
    }
    return mst_cost;
}
```

**Python:**

```python
def kruskal(edges, n):
    edges.sort(key=lambda x: x[2])
    dsu = DSU(n)
    mst_cost = 0
    for u, v, w in edges:
        if dsu.unite(u, v):
            mst_cost += w
    return mst_cost
```

## 7. Topological Sort (Kahn's Algorithm)

**C++:**

```cpp
vector<int> topological_sort(vector<vector<int>>& adj, vector<int>& indeg, int n) {
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indeg[i] == 0) q.push(i);
    }
    vector<int> order;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);
        for (int v : adj[u]) {
            if (--indeg[v] == 0) q.push(v);
        }
    }
    return order;
}
```

**Python:**

```python
from collections import deque

def topological_sort(adj, indeg, n):
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order
```

## 8. Longest Common Subsequence (LCS)

**C++:**

```cpp
int lcs(string& s1, string& s2) {
    int m = s1.size(), n = s2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
```

**Python:**

```python
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

## 9. Knapsack (0/1)

**C++:**

```cpp
int knapsack(vector<int>& wt, vector<int>& val, int W) {
    int n = wt.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (wt[i-1] <= w) {
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt[i-1]] + val[i-1]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    return dp[n][W];
}
```

**Python:**

```python
def knapsack(wt, val, W):
    n = len(wt)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if wt[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt[i-1]] + val[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]
```

## 10. Kadane's Algorithm (Max Subarray Sum)

**C++:**

```cpp
long long kadane(vector<int>& arr) {
    long long max_sum = arr[0], curr = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        curr = max((long long)arr[i], curr + arr[i]);
        max_sum = max(max_sum, curr);
    }
    return max_sum;
}
```

**Python:**

```python
def kadane(arr):
    max_sum = curr = arr[0]
    for i in range(1, len(arr)):
        curr = max(arr[i], curr + arr[i])
        max_sum = max(max_sum, curr)
    return max_sum
```

## 11. Sieve of Eratosthenes

**C++:**

```cpp
vector<bool> sieve(int n) {
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }
    return is_prime;
}
```

**Python:**

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return is_prime
```

## 12. Binary Exponentiation

**C++:**

```cpp
long long binpow(long long a, long long b, long long mod) {
    long long res = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}
```

**Python:**

```python
def binpow(a, b, mod):
    res = 1
    a %= mod
    while b > 0:
        if b & 1:
            res = (res * a) % mod
        a = (a * a) % mod
        b >>= 1
    return res
```

## 13. GCD and LCM

**C++:**

```cpp
long long gcd(long long a, long long b) {
    return b ? gcd(b, a % b) : a;
}

long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b;
}
```

**Python:**

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a // gcd(a, b) * b
```

## 14. Segment Tree (Range Sum Query)

**C++:**

```cpp
class SegTree {
    vector<long long> tree;
    int n;
    void build(vector<int>& arr, int node, int l, int r) {
        if (l == r) {
            tree[node] = arr[l];
            return;
        }
        int m = (l + r) / 2;
        build(arr, 2*node, l, m);
        build(arr, 2*node+1, m+1, r);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
    long long query(int node, int l, int r, int ql, int qr) {
        if (ql > r || qr < l) return 0;
        if (ql <= l && r <= qr) return tree[node];
        int m = (l + r) / 2;
        return query(2*node, l, m, ql, qr) + query(2*node+1, m+1, r, ql, qr);
    }
    void update(int node, int l, int r, int pos, int val) {
        if (l == r) {
            tree[node] = val;
            return;
        }
        int m = (l + r) / 2;
        if (pos <= m) update(2*node, l, m, pos, val);
        else update(2*node+1, m+1, r, pos, val);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
public:
    SegTree(vector<int>& arr) : n(arr.size()), tree(4*n) {
        build(arr, 1, 0, n-1);
    }
    long long query(int l, int r) { return query(1, 0, n-1, l, r); }
    void update(int pos, int val) { update(1, 0, n-1, pos, val); }
};
```

**Python:**

```python
class SegTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, l, r):
        if l == r:
            self.tree[node] = arr[l]
            return
        m = (l + r) // 2
        self._build(arr, 2*node, l, m)
        self._build(arr, 2*node+1, m+1, r)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def query(self, ql, qr):
        return self._query(1, 0, self.n-1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if ql > r or qr < l:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        m = (l + r) // 2
        return self._query(2*node, l, m, ql, qr) + self._query(2*node+1, m+1, r, ql, qr)

    def update(self, pos, val):
        self._update(1, 0, self.n-1, pos, val)

    def _update(self, node, l, r, pos, val):
        if l == r:
            self.tree[node] = val
            return
        m = (l + r) // 2
        if pos <= m:
            self._update(2*node, l, m, pos, val)
        else:
            self._update(2*node+1, m+1, r, pos, val)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]
```

## 15. Fenwick Tree (Binary Indexed Tree)

**C++:**

```cpp
class FenwickTree {
    vector<long long> bit;
    int n;
public:
    FenwickTree(int n) : n(n), bit(n + 1, 0) {}
    void update(int idx, int delta) {
        for (++idx; idx <= n; idx += idx & -idx)
            bit[idx] += delta;
    }
    long long query(int idx) {
        long long sum = 0;
        for (++idx; idx > 0; idx -= idx & -idx)
            sum += bit[idx];
        return sum;
    }
    long long range_query(int l, int r) {
        return query(r) - (l > 0 ? query(l - 1) : 0);
    }
};
```

**Python:**

```python
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n + 1)

    def update(self, idx, delta):
        idx += 1
        while idx <= self.n:
            self.bit[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        s = 0
        idx += 1
        while idx > 0:
            s += self.bit[idx]
            idx -= idx & -idx
        return s

    def range_query(self, l, r):
        return self.query(r) - (self.query(l - 1) if l > 0 else 0)
```

## 16. Trie (Prefix Tree)

**C++:**

```cpp
class Trie {
    struct Node {
        map<char, Node*> children;
        bool is_end = false;
    };
    Node* root;
public:
    Trie() : root(new Node()) {}
    void insert(string& word) {
        Node* curr = root;
        for (char c : word) {
            if (!curr->children[c]) curr->children[c] = new Node();
            curr = curr->children[c];
        }
        curr->is_end = true;
    }
    bool search(string& word) {
        Node* curr = root;
        for (char c : word) {
            if (!curr->children[c]) return false;
            curr = curr->children[c];
        }
        return curr->is_end;
    }
};
```

**Python:**

```python
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        curr = self.root
        for c in word:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]
        curr['#'] = True

    def search(self, word):
        curr = self.root
        for c in word:
            if c not in curr:
                return False
            curr = curr[c]
        return '#' in curr
```

## 17. KMP String Matching

**C++:**

```cpp
vector<int> kmp_search(string& text, string& pattern) {
    int n = text.size(), m = pattern.size();
    vector<int> lps(m, 0);
    for (int i = 1, len = 0; i < m; ) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        } else if (len) {
            len = lps[len - 1];
        } else {
            i++;
        }
    }
    vector<int> matches;
    for (int i = 0, j = 0; i < n; ) {
        if (text[i] == pattern[j]) {
            i++; j++;
        }
        if (j == m) {
            matches.push_back(i - j);
            j = lps[j - 1];
        } else if (i < n && text[i] != pattern[j]) {
            if (j) j = lps[j - 1];
            else i++;
        }
    }
    return matches;
}
```

**Python:**

```python
def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    lps = [0] * m
    i, length = 1, 0
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length:
            length = lps[length - 1]
        else:
            i += 1

    matches = []
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j:
                j = lps[j - 1]
            else:
                i += 1
    return matches
```

## 18. Floyd-Warshall (All Pairs Shortest Path)

**C++:**

```cpp
void floyd_warshall(vector<vector<int>>& dist, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
}
```

**Python:**

```python
def floyd_warshall(dist, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
```

## 19. Bellman-Ford (Negative Weight Shortest Path)

**C++:**

```cpp
vector<int> bellman_ford(int src, vector<tuple<int,int,int>>& edges, int n) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    for (int i = 0; i < n - 1; i++) {
        for (auto& [u, v, w] : edges) {
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
    return dist;
}
```

**Python:**

```python
def bellman_ford(src, edges, n):
    dist = [float('inf')] * n
    dist[src] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist
```

## 20. Lowest Common Ancestor (Binary Lifting)

**C++:**

```cpp
class LCA {
    vector<vector<int>> up;
    vector<int> depth;
    int LOG;
    void dfs(int v, int p, vector<vector<int>>& adj) {
        up[v][0] = p;
        for (int i = 1; i < LOG; i++) {
            up[v][i] = up[up[v][i-1]][i-1];
        }
        for (int u : adj[v]) {
            if (u != p) {
                depth[u] = depth[v] + 1;
                dfs(u, v, adj);
            }
        }
    }
public:
    LCA(int n, vector<vector<int>>& adj, int root = 0) {
        LOG = ceil(log2(n)) + 1;
        up.assign(n, vector<int>(LOG, 0));
        depth.assign(n, 0);
        dfs(root, root, adj);
    }
    int lca(int u, int v) {
        if (depth[u] < depth[v]) swap(u, v);
        int diff = depth[u] - depth[v];
        for (int i = 0; i < LOG; i++) {
            if ((diff >> i) & 1) u = up[u][i];
        }
        if (u == v) return u;
        for (int i = LOG - 1; i >= 0; i--) {
            if (up[u][i] != up[v][i]) {
                u = up[u][i];
                v = up[v][i];
            }
        }
        return up[u][0];
    }
};
```

**Python:**

```python
import math

class LCA:
    def __init__(self, n, adj, root=0):
        self.LOG = math.ceil(math.log2(n)) + 1
        self.up = [[0] * self.LOG for _ in range(n)]
        self.depth = [0] * n
        self._dfs(root, root, adj)

    def _dfs(self, v, p, adj):
        self.up[v][0] = p
        for i in range(1, self.LOG):
            self.up[v][i] = self.up[self.up[v][i-1]][i-1]
        for u in adj[v]:
            if u != p:
                self.depth[u] = self.depth[v] + 1
                self._dfs(u, v, adj)

    def lca(self, u, v):
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = self.depth[u] - self.depth[v]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u = self.up[u][i]
        if u == v:
            return u
        for i in range(self.LOG - 1, -1, -1):
            if self.up[u][i] != self.up[v][i]:
                u = self.up[u][i]
                v = self.up[v][i]
        return self.up[u][0]
```

## Quick Tips

- Always check constraints and choose appropriate data types
- Use `long long` in C++ for large numbers
- Watch out for integer overflow
- Remember to initialize arrays properly
- Test edge cases: empty input, single element, all same elements
- For Python: use `sys.stdin.readline()` for faster I/O in large inputs
