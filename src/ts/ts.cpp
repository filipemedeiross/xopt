#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include "../pmedian/instance.h"
#include "../pmedian/solution.h"
#include "../pmedian/evaluate.h"
#include "ts.h"

#define UPPERB  1e18
#define EPSILON 1e-9

using namespace std;

Solution tspmed (const Instance& instance, const vector <int>& medoids, int iter_factor) {
    int    iter, last;
    double best_cost, cur_cost;

    int n = instance.get_n();
    int p = instance.get_p();
    mt19937 rng (random_device{}());

    int tenure = (int) p / 2;
    int max_iter    = iter_factor * n;
    int stable_iter = 0.3 * max_iter;

    vector <int> S = medoids;
    vector <int> best_S = S;

    Evaluation eval = evaluate(instance, S);
    cur_cost = best_cost = eval.cost;

    vector <int>  time  (n, -p   );
    vector <bool> in_solution(n, false);
    for (int vi : S)
        in_solution[vi] = true;

    iter = last = 0;
    while ((iter - last) < stable_iter && iter < max_iter) {
        int best_out = -1;
        int best_in  = -1;
        double best_move_cost = UPPERB;
        bool   best_is_tabu = false;

        for (int idx = 0; idx < p; ++idx) {
            int out_vertex = S[idx];
            bool out_tabu = (iter - time[out_vertex]) < tenure;

            for (int v = 0; v < n; ++v) {
                if (in_solution[v])
                    continue;

                bool in_tabu = (iter - time[v]) < tenure;
                double move_cost = evaluate_interchange(
                    instance,
                    eval,
                    out_vertex,
                    v
                );

                bool tabu = out_tabu || in_tabu;
                bool aspiration = move_cost + EPSILON < best_cost;

                if ((tabu && !aspiration))
                    continue;

                if (move_cost + EPSILON < best_move_cost ||
                    (abs(move_cost - best_move_cost) <= EPSILON && tabu && !best_is_tabu)) {
                    best_move_cost = move_cost;
                    best_out = idx;
                    best_in  = v;
                    best_is_tabu = tabu && !aspiration;
                }
            }
        }

        if (best_out == -1 || best_in == -1)
            break;

        int removed = S[best_out];
        in_solution[removed] = false;
        in_solution[best_in] = true;
        S[best_out] = best_in;

        time[removed] = iter;
        time[best_in] = iter;

        eval = evaluate(instance, S);
        cur_cost = eval.cost;

        if (cur_cost + EPSILON < best_cost) {
            best_S    = S;
            best_cost = cur_cost;
            last      = iter;
        }

        if ((iter - last + 1) % ((int) stable_iter / 10) == 0)
            tenure = uniform_int_distribution <> ((int) p / 2, p - 1) (rng);

        iter++;
    }

    return {best_cost, best_S};
}
