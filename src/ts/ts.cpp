#include <vector>
#include <random>
#include <algorithm>

#include "../pmedian/instance.h"
#include "../pmedian/solution.h"
#include "../pmedian/evaluate.h"
#include "ts.h"

#define UPPERB  1e18

using namespace std;

bool choose_move (vector <int>& S, int p, int slack) {
    mt19937 rng (random_device{}());
    uniform_real_distribution<> coin (0.0, 1.0);

    if ((int) S.size() < p - slack)
        return true ;
    if ((int) S.size() > p + slack)
        return false;

    return coin(rng) > 0.5;
}

Solution tspmed (const Instance& instance, const vector <int>& medoids, int iter_factor) {
    bool   tabu;
    int    v, iter, last, node;
    double c, best_cost, cur_cost, move_cost;

    int n = instance.get_n();
    int p = instance.get_p();
    mt19937 rng (random_device{}());

    int slack  = 0;
    int tenure = (int) p / 2;
    int max_iter    = iter_factor * n;
    int stable_iter = 0.3 * max_iter;

    vector <int> S = medoids;
    vector <int> best_S = S;

    cur_cost = best_cost = evaluate(instance, S);

    vector <int>  time  (n, -p   );
    vector <bool> added (n, false);
    for (int vi : S)
        added[vi] = true;

    iter = last = 0;
    while ((iter - last) < stable_iter && iter < max_iter) {
        node = -1;
        move_cost = UPPERB;

        if (choose_move(S, p, slack)) {
            for (v = 0; v < n; v++) {
                if (added[v] ||
                   (iter - time[v]) < tenure)
                    continue;

                S.push_back(v);
                c = evaluate(instance, S);
                S.pop_back ( );

                if (c < move_cost) {
                    node      = v;
                    move_cost = c;
                }
            }

            S.push_back(node);
            added[node] = true;
            time [node] = iter;
        } else {
            for (size_t i = 0; i < S.size(); ++i) {
                v = S[i];

                swap(S[i], S.back());
                S.pop_back();

                c    = evaluate(instance, S);
                tabu = (iter - time[v]) < tenure;

                if (( tabu && c < best_cost) ||
                    (!tabu && c < move_cost)) {
                    node      = v;
                    move_cost = c;
                }

                S.push_back(v);
                swap(S[i], S.back());
            }

            S.erase(
                find(S.begin(), S.end(), node)
            );
            added[node] = false;
        }

        cur_cost = move_cost;
        if ((int) S.size() == p && cur_cost < best_cost) {
            best_S    = S;
            best_cost = cur_cost;

            slack = 0;
            last  = iter;
        }

        if ((iter - last + 1) % ((int) stable_iter / 10) == 0)
            tenure = uniform_int_distribution <> ((int) p / 2, p - 1) (rng);
        if ((iter - last + 1) % ((int) stable_iter /  2) == 0)
            slack++;

        iter++;
    }

    return {best_cost, best_S};
}