#include <vector>
#include <numeric>
#include <algorithm>

#include "instance.h"
#include "evaluate.h"

using namespace std;

Evaluation evaluate (const Instance& instance, const vector <int>& medoids) {
    int dist, u, n = instance.get_n();

    Evaluation eval;

    eval.closest.assign(n, -1);
    eval.second .assign(n, -1);
    eval.d1     .assign(n, Instance::INF);
    eval.d2     .assign(n, Instance::INF);

    for (int medoid : medoids) {
        for (u = 0; u < n; u++) {
            dist = instance(u, medoid);

            if (dist < eval.d1[u]) {
                eval.d2[u] = eval.d1[u];
                eval.d1[u] = dist;

                eval.second [u] = eval.closest[u];
                eval.closest[u] = medoid;
            } else if (dist < eval.d2[u]) {
                eval.d2    [u] = dist;
                eval.second[u] = medoid;
            }
        }
    }

    eval.cost = accumulate(eval.d1.begin(), eval.d1.end(), 0.0);

    return eval;
}

double evaluate_cost (const Instance& instance, const vector <int>& medoids) {
    return evaluate(instance, medoids).cost;
}

double evaluate_swap (const Instance& instance, const Evaluation & eval, int out, int in) {
    int cur, upd;
    size_t  u, n;
    double new_cost = eval.cost;

    n = eval.closest.size();

    for (u = 0; u < n; u++) {
        cur = eval.d1[u];
        upd = min(
            instance(u, in),
            (eval.closest[u] == out) ? eval.d2[u] : cur
        );

        new_cost += static_cast<double> (upd - cur);
    }

    return new_cost;
}