#include <vector>
#include <random>
#include <algorithm>
#include <execution>
#include <unordered_set>

#include "../pmedian/instance.h"
#include "../pmedian/evaluate.h"
#include "kmedoids.h"
#include "medoids.h"

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
    bool improved;
    int  iter;

    int n = instance.get_n();
    int p = instance.get_p();

    unordered_set<int> medoid_set(
        sol.medoids.begin(),
        sol.medoids.end  ()
    );

    iter = 0;
    do {
        iter++;
        improved = false;

        for (int i = 0; i < p; ++i) {
            int current = sol.medoids[i];
            double best_cost = sol.cost;
            int best_candidate = current;

            for (int h = 0; h < n; ++h) {
                if (medoid_set.contains(h))
                    continue;

                double new_cost = evaluate_interchange(
                    instance,
                    sol.eval,
                    current,
                    h
                );

                if (new_cost + EPSILON < best_cost) {
                    best_cost = new_cost;
                    best_candidate = h;
                }
            }

            if (best_candidate != current) {
                medoid_set.erase(current);
                medoid_set.insert(best_candidate);

                sol.medoids[i] = best_candidate;
                sol.eval  = evaluate(instance, sol.medoids);
                sol.cost  = sol.eval.cost;

                improved = true;
            }
        }
    } while (improved && iter < max_iter);

    return sol;
}

vector <Medoids> kmedoids(const Instance& instance, int max_iter, int n_restarts) {
    int restart;

    int n = instance.get_n();
    int p = instance.get_p();
    mt19937 rng (random_device{}());

    vector <int>     usage  (n, 0  );
    vector <double > weights(n, 1.0);
    vector <Medoids> all_solutions  ;
    all_solutions.reserve(n_restarts);

    for (restart = 0; restart < n_restarts; restart++) {
        vector <int> medoids = initialize_medoids(p, usage, weights, rng);

        Evaluation eval = evaluate(instance, medoids);
        all_solutions.push_back({
            eval.cost,
            medoids,
            eval
        });
    }

    for_each (
        execution::par,
        all_solutions.begin(),
        all_solutions.end  (),
        [&](Medoids& sol) {
            sol = local_search(instance, sol, max_iter);
        }
    );

    sort (
        all_solutions.begin(),
        all_solutions.end  (),
        [](const Medoids& a, const Medoids& b) {
            return a.cost < b.cost;
        }
    );

    return all_solutions;
}