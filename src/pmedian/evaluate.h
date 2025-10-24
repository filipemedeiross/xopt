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
};

Evaluation evaluate  (const Instance&, const vector <int>&);

double evaluate_cost (const Instance&, const vector <int>&);
double evaluate_swap (const Instance&, const Evaluation  &, int, int);

#endif