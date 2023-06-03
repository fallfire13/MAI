#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

class TSegment {
public:
    int l;
    int r;
    int i;

    TSegment &operator=(const TSegment &rhs) = default;
};

bool Cmp(TSegment &lhs, TSegment &rhs) {
    return (lhs.l < rhs.l);
}

bool CmpIndex(TSegment &lhs, TSegment &rhs) {
    return (lhs.i < rhs.i);
}

void SegmentSelection(vector<TSegment> &segments, int m) {
    vector<TSegment> ans;

    sort(segments.begin(), segments.end(), Cmp);

    int i = 0;
    int lastCorrectSegment = -1;

    for (; i < segments.size(); ++i) {
        if (segments[i].l <= 0 && segments[i].r >= 0) {
            lastCorrectSegment = i;
        } else if (segments[i].l > 0) {
            break;
        }
    }

    if (lastCorrectSegment == -1) {
        cout << 0 << endl;
        return;
    }

    ans.push_back(segments[lastCorrectSegment]);

    bool complete = false;
    lastCorrectSegment = -1;
    while (i < segments.size()) {
        for (; i < segments.size(); ++i) {
            if (segments[i].l <= ans.back().r && segments[i].r >= ans.back().r) {
                lastCorrectSegment = i;
                if (segments[i].r >= m) {
                    complete = true;
                    break;
                }

            } else if (segments[i].l > ans.back().r) {
                break;
            }
        }

        if (lastCorrectSegment == -1) {
            cout << 0 << endl;
            return;
        }

        ans.push_back(segments[lastCorrectSegment]);
        if (m <= ans.back().r || complete) {
            break;
        }

        lastCorrectSegment = -1;
    }


    if (m <= ans.back().r) {
        cout << ans.size() << endl;
        sort(ans.begin(), ans.end(), CmpIndex);
        for (auto j : ans) {
            cout << j.l << " " << j.r << endl;
        }
    } else {
        cout << 0 << endl;
    }
}

int main() {
    size_t n;
    cin >> n;

    vector<TSegment> segments(n);

    for (int i = 0; i < n; ++i) {
        segments[i].i = i;
        cin >> segments[i].l >> segments[i].r;
    }

    int m;
    cin >> m;

    SegmentSelection(segments, m);

    return 0;
}
