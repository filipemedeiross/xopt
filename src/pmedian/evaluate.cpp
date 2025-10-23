#include <vector>
#include <numeric>
#include <limits>
#include <algorithm>
#include "instance.h"
#include "evaluate.h"

using namespace std;

static constexpr int INF_DISTANCE = numeric_limits<int>::max();

Evaluation evaluate (const Instance& instance, const vector<int>& S) {
    int n = instance.get_n();

    Evaluation eval;
    eval.cost            = 0.0;
    eval.best_medoid     = vector<int>(n, -1);
    eval.second_medoid   = vector<int>(n, -1);
    eval.best_distance   = vector<int>(n, INF_DISTANCE);
    eval.second_distance = vector<int>(n, INF_DISTANCE);

    for (int medoid : S) {
        for (int i = 0; i < n; ++i) {
            int dist = instance(i, medoid);

            if (dist < eval.best_distance[i]) {
                eval.second_distance[i] = eval.best_distance[i];
                eval.second_medoid  [i] = eval.best_medoid[i];

                eval.best_distance[i] = dist;
                eval.best_medoid  [i] = medoid;
            } else if (dist < eval.second_distance[i]) {
                eval.second_distance[i] = dist;
                eval.second_medoid  [i] = medoid;
            }
        }
    }

    for (int d : eval.best_distance)
        eval.cost += d == INF_DISTANCE ? 0.0 : static_cast<double>(d);

    return eval;
}

double evaluate_interchange(
    const Instance& instance,
    const Evaluation& eval,
    int removed_medoid,
    int candidate_medoid
) {
    int n = instance.get_n();
    double total = 0.0;

    for (int i = 0; i < n; ++i) {
        int candidate_distance = instance(i, candidate_medoid);
        int best_distance      = eval.best_distance[i];
        int second_distance    = eval.second_distance[i];

        if (eval.best_medoid[i] == removed_medoid) {
            int fallback = second_distance == INF_DISTANCE ? Instance::INF : second_distance;
            total += min(fallback, candidate_distance);
        } else {
            total += min(best_distance, candidate_distance);
        }
    }

    return total;
}
