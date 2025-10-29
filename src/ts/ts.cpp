#include <vector>
#include <random>
#include <limits>
#include <memory>
#include <algorithm>

#include "../pmedian/instance.h"
#include "../pmedian/solution.h"
#include "../pmedian/evaluate.h"
#include "../tools/trie.h"
#include "ts.h"

using namespace std;

constexpr double EPSILON = 1e-9;

TSResult tspmed (
    const Instance&           instance   ,
    const vector <int>&       medoids    ,
    int                       iter_factor,
    shared_ptr <SolutionTrie> long_term
) {
    int    iter, last;
    int    idx, out, in;
    int    best_in, best_out;
    double best_cost, candidate_cost;
    double best_move_cost, best_tabu_cost;

    int n = instance.get_n();
    int p = instance.get_p();
    mt19937 rng (random_device{}());

    int max_iter    = iter_factor * n;
    int stable_iter = static_cast <int> (0.3 * max_iter);

    int tenure      = static_cast <int> (p / 2);
    int span_tenure = static_cast <int> (stable_iter / 5);

    vector <int> S      = medoids;
    vector <int> best_S = S;

    Evaluation eval = evaluate(instance, S);
    best_cost       = eval.cost;

    TSResult result (long_term ?
                     long_term :
                     make_shared <SolutionTrie> (n, p));

    vector <int>              time (n, -p);
    shared_ptr <SolutionTrie> memory = result.long_term;

    vector <bool> in_solution(n, false);
    for (int v : S)
        in_solution[v] = true;

    last = iter = 0;
    while ((iter - last) < stable_iter && iter < max_iter) {
        best_tabu_cost = numeric_limits <double>::infinity();
        best_move_cost = numeric_limits <double>::infinity();
        best_in  = -1;
        best_out = -1;

        for (idx = 0; idx < p; idx++) {
            out = S[idx];

            for (in = 0; in < n; in++) {
                if (in_solution[in])
                    continue;
                if (memory->contains_swap(in_solution, out, in))
                    continue;

                candidate_cost = evaluate_swap(instance, eval, out, in);

                if ((iter - time[in]) < tenure) {
                    if (candidate_cost + EPSILON < best_cost &&
                        candidate_cost + EPSILON < best_move_cost) {
                        best_move_cost = candidate_cost;
                        best_in  = in ;
                        best_out = out;
                    } else if (candidate_cost + EPSILON < best_tabu_cost) {
                        best_tabu_cost = candidate_cost;
                    }
                } else if (candidate_cost + EPSILON < best_move_cost) {
                    best_move_cost = candidate_cost;
                    best_in  = in ;
                    best_out = out;
                }
            }
        }

        if (best_move_cost >= eval.cost &&
            best_tabu_cost >= eval.cost)
            memory->update(S);

        *ranges::find(S, best_out) = best_in;

        update_evaluation (instance,
                           S       ,
                           best_out,
                           best_in ,
                           eval);
        if (eval.cost + EPSILON < best_cost) {
            best_S    = S;
            best_cost = eval.cost;

            last = iter;
        }

        in_solution[best_out] = false;
        in_solution[best_in ] = true;
        time       [best_out] = iter;

        if ((iter - last + 1) % span_tenure == 0)
            tenure = uniform_int_distribution <int> ((int) p / 2, p - 1)(rng);

        iter++;
    }

    result.best = {best_cost,
                   best_S   };

    return result;
}