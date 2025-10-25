#include <vector>
#include <random>
#include <algorithm>
#include <execution>
#include <unordered_set>

#include "../pmedian/instance.h"
#include "../pmedian/evaluate.h"
#include "medoids.h"
#include "kmedoids.h"

#define EPSILON 1e-9

using namespace std;

vector <int> initialize_medoids (
    int p,
    vector <int>&    usage  ,
    vector <double>& weights,
    mt19937& rng
) {
    int i, m;

    vector <int>        medoids(p);
    unordered_set <int> medoid_set;
    medoid_set.reserve(p);

    discrete_distribution <int> dist(
        weights.begin(),
        weights.end  ()
    );

    for (i = 0; i < p; i++) {
        do {
            m = dist(rng);
        } while (medoid_set.contains(m));

        medoids[i] = m;
        medoid_set.insert(m);

        usage  [m]++;
        weights[m] = 1.0 / (1.0 + usage[m]);
    }

    return medoids;
}

Medoids local_search (const Instance& instance, Medoids sol, int max_iter) {
    bool   improved;
    int    i, h, iter;
    int    old, candidate;
    double best_cost, candidate_cost;

    int n = instance.get_n();
    int p = instance.get_p();

    unordered_set <int> medoid_set(
        sol.medoids.begin(),
        sol.medoids.end  ()
    );
    Evaluation eval = evaluate(instance, sol.medoids);

    sol.cost = eval.cost;

    iter = 0;
    do {
        iter++;
        improved = false;

        for (i = 0; i < p; i++) {
            old = sol.medoids[i];
            best_cost = sol.cost;

            candidate = -1;
            for (h = 0; h < n; ++h) {
                if (medoid_set.contains(h))
                    continue;

                candidate_cost = evaluate_swap(instance, eval, old, h);

                if (candidate_cost + EPSILON < best_cost) {
                    candidate = h;
                    best_cost = candidate_cost;
                }
            }

            if (candidate != -1) {
                sol.medoids[i] = candidate;
                sol.cost       = best_cost;

                medoid_set.erase  (old);
                medoid_set.insert (candidate);
                update_evaluation (instance, sol.medoids, old, candidate, eval);

                improved = true;
            }
        }
    } while (improved && iter < max_iter);

    return sol;
}

vector <Medoids> kmedoids (const Instance& instance, int max_iter, int restarts) {
    int r;
    int n = instance.get_n();
    int p = instance.get_p();

    mt19937 rng (random_device{}());

    vector <int>     usage     (n, 0  );
    vector <double > weights   (n, 1.0);
    vector <Medoids> solutions (restarts);

    for (r = 0; r < restarts; r++)
        solutions[r] = {0.0, initialize_medoids(p, usage, weights, rng)};

    for_each (
        execution::par,
        solutions.begin(),
        solutions.end  (),
        [&](Medoids& sol) {
            sol = local_search(instance, sol, max_iter);
        }
    );

    sort (
        solutions.begin(),
        solutions.end  (),
        [](const Medoids& a, const Medoids& b) {
            return a.cost < b.cost;
        }
    );

    return solutions;
}