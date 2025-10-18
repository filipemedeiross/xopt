#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <unordered_set>

#include "../pmedian/instance.h"
#include "../pmedian/evaluate.h"
#include "kmedoids.h"

#define UPPERB  1e18
#define EPSILON 1e-9

vector <int> kmedoids (const Instance& instance, int max_iter, int n_restarts) {
    bool   improved;
    int    old, i, h;
    int    restart, iter;
    double best_cost, cur_cost, new_cost;

    int n = instance.get_n();
    int p = instance.get_p();
    mt19937 rng (random_device{}());

    vector<int> nodes        (n);
    vector<int> medoids      (p);
    vector<int> best_medoids (p);
    unordered_set<int> medoid_set;

    iota (nodes.begin(), nodes.end(), 0);
    medoid_set.reserve(p);

    best_cost = UPPERB;
    for (restart = 0; restart < n_restarts; restart++) {
        shuffle(nodes.begin(), nodes.end(), rng);

        medoid_set.clear();
        for (i = 0; i < p; ++i) {
            medoids[i] = nodes[i];
            medoid_set.insert(nodes[i]);
        }

        iter     = 0;
        cur_cost = evaluate(instance, medoids);
        do {
            improved = false;
            iter++;

            for (i = 0; i < p; i++) {
                old = medoids[i];

                for (h = 0; h < n; h++) {
                    if (medoid_set.count(h))
                        continue;

                    medoids[i] = h;
                    new_cost   = evaluate(instance, medoids);

                    if (new_cost + EPSILON < cur_cost) {
                        cur_cost = new_cost;
                        improved = true;

                        medoid_set.erase (old);
                        medoid_set.insert(h  );
                        old = h;
                    } else {
                        medoids[i] = old;
                    }
                }
            }
        } while (improved && iter < max_iter);

        if (cur_cost < best_cost) {
            best_cost    = cur_cost;
            best_medoids = medoids ;
        }
    }

    return best_medoids;
}