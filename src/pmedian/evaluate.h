#ifndef EVALUATE_H
#define EVALUATE_H

#include <vector>
#include "instance.h"

struct Evaluation {
    double cost;
    std::vector<int> best_medoid;
    std::vector<int> second_medoid;
    std::vector<int> best_distance;
    std::vector<int> second_distance;
};

Evaluation evaluate (const Instance&, const std::vector<int>&);
double evaluate_interchange(
    const Instance&,
    const Evaluation&,
    int removed_medoid,
    int candidate_medoid
);

#endif
