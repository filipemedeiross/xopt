#ifndef TS_H
#define TS_H

#include <vector>
#include "instance.h"
#include "solution.h"

std::vector <int> kmedoids (const Instance&, int = 50);

double   evaluate (const Instance&, const std::vector <int>&);
Solution tspmed   (const Instance&, int = 2);

#endif