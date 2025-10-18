#include <vector>
#include <numeric>
#include "instance.h"
#include "evaluate.h"

double evaluate (const Instance& instance, const vector <int>& S) {
    int i, n = instance.get_n();

    vector <int> costs (n, Instance::INF);
    for (i = 0; i < n; i++)
        for (int f : S)
            costs[i] = min(costs[i], instance(i, f));

    return accumulate(costs.begin(), costs.end(), 0.0);
}