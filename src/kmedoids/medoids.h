#ifndef MEDOIDS_H
#define MEDOIDS_H

#include <vector>
#include "../pmedian/evaluate.h"

struct Medoids {
    double cost;
    std::vector<int> medoids;
    Evaluation eval;
};

#endif
