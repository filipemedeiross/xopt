#ifndef EVALUATE_H
#define EVALUATE_H

#include <vector>
#include "instance.h"

using namespace std;

struct Evaluation {
    double cost = 0.0;

    vector <int> closest;
    vector <int> second ;
    vector <int> d1;
    vector <int> d2;

    void update_first  (int u, int medoid, int dist) {
        second [u] = closest[u];
        closest[u] = medoid;
        d2[u] = d1[u];
        d1[u] = dist;
    }

    void update_second (int u, int medoid, int dist) {
        second[u] = medoid;
        d2[u] = dist;
    }

    void recompute_second (const Instance& instance, const vector <int>& medoids, int u) {
        int dist, best = closest[u];

        int second    = -1;
        int best_dist = Instance::INF;

        for (int medoid : medoids) {
            if (medoid == best)
                continue;

            dist = instance(u, medoid);
            if (dist < best_dist) {
                second    = medoid;
                best_dist = dist;
            }
        }

        update_second(u, second, best_dist);
    }
};

Evaluation evaluate (const Instance&, const vector <int>&);

double evaluate_cost     (const Instance&, const vector <int>&);
double evaluate_swap     (const Instance&, const Evaluation  &, int, int);
void   update_evaluation (const Instance&, const vector <int>&, int, int, Evaluation&);

#endif