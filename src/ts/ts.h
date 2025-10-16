#ifndef TS_H
#define TS_H

#include <vector>
#include "instance.h"

using namespace std;

struct Solution {
    double cost;
    vector <int> facilities;
};

vector <int> kmedoids (const Instance&, int = 50);

double   evaluate (const Instance&, const vector <int>&);
Solution tspmed   (const Instance&, int = 2);

#endif