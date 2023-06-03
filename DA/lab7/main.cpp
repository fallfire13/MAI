#include <iostream>
#include <map>
#include <stack>
#include <vector>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int n, m;
    cin >> n >> m;

    if (n == 0) {
        cout << 0 << endl;
        return 0;
    }

    vector<int> w(n), c(n);
    for (int it = 0; it < n; ++it) cin >> w[it] >> c[it];

    vector<vector<vector<long long>>> dp(n + 1);
    for (int it = 0; it <= n; ++it) {
        dp[it].assign(m + 1, vector<long long>(it + 1, -1));
        for (auto& j_it : dp[it]) j_it[0] = 0;
    }

    for (int viewed = 0; viewed < n; ++viewed) {
        for (int weight = 0; weight <= m; ++weight) {
            for (int picked = 0; picked <= viewed; ++picked) {
                int sum_weight = weight + w[viewed];
                if (sum_weight <= m) {
                    if (dp[viewed][weight][picked] != -1) {
                        dp[viewed + 1][sum_weight][picked + 1] =
                            dp[viewed][weight][picked] + c[viewed];
                    }
                }
                dp[viewed + 1][weight][picked] =
                    max(dp[viewed + 1][weight][picked],
                             dp[viewed][weight][picked]);
            }
        }
    }

    long long ans = 0, mark = -1;
    for (int it = 0; it <= n; ++it) {
        long long tmp = it * dp[n][m][it];
        if (tmp > ans) {
            mark = it;
            ans = tmp;
        }
    }
    cout << ans << endl;

    stack<int> path;
    if (mark != -1) {
        int item = n, weight = m;
        while (item) {
            if (dp[item][weight][mark] == 0) {
                break;
            }
            if (mark == item || dp[item][weight][mark] !=
                                    dp[item - 1][weight][mark]) {
                --mark;
                weight -= w[item - 1];
                path.push(item);
            }

            --item;
        }
    }

    if (!path.empty()) {
        while (!path.empty()) {
            cout << path.top();
            path.pop();
            if (!path.empty()) cout << ' ';
        }
        cout << endl;
    }
    return 0;
}
