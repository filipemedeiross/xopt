#include <bits/stdc++.h>
#include <iostream>
#include <unordered_set>
#include "instance.h"
#include "solution.h"
#include "ts.h"

#define UPPERB  1e18
#define EPSILON 1e-9

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

double evaluate (const Instance& instance, const vector <int>& S) {
    int i, n = instance.get_n();

    vector <int> costs (n, Instance::INF);
    for (i = 0; i < n; i++)
        for (int f : S)
            costs[i] = min(costs[i], instance(i, f));

    return accumulate(costs.begin(), costs.end(), 0.0);
}

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

Solution tspmed (const Instance& instance, int iter_factor) {
    bool   tabu;
    int    v, iter, last, noimprove, node;
    double c, best_cost, cur_cost, move_cost;

    int n = instance.get_n();
    int p = instance.get_p();
    mt19937 rng (random_device{}());

    int slack  = 0;
    int tenure = (int) p / 2;
    int max_iter    = iter_factor * n;
    int stable_iter = 0.2 * max_iter;

    vector <int> S = kmedoids(instance);
    vector <int> best_S = S;

    cur_cost = best_cost = evaluate(instance, S);

    vector <int>  time  (n, -p   );
    vector <bool> added (n, false);
    for (int vi : S)
        added[vi] = true;

    iter = last = noimprove = 0;
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

        if (move_cost < cur_cost)
            noimprove = 0;
        else
            noimprove++;
        cur_cost = move_cost;

        if (S.size() == p && cur_cost < best_cost) {
            best_S    = S;
            best_cost = cur_cost;

            slack = 0;
            last  = iter;
        }

        if ((noimprove + 1  ) % ((int) stable_iter / 10) == 0)
            tenure = uniform_int_distribution <> (1, p - 1)(rng);
        if ((noimprove + 1  ) % ((int) stable_iter /  5) == 0)
            slack++;
        /*if ((iter - last + 1) % ((int) stable_iter /  2) == 0) {
            fill(added.begin(), added.end(), false);

            cout << "Restarting at iteration " << iter << endl;
            S = kmedoids(instance);
            for (int vi : S)
                added[vi] = true;

            cur_cost = evaluate(instance, S);
        }*/

        iter++;
    }

    return {best_cost, best_S};
}