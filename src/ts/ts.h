#ifndef TS_H
#define TS_H

#include <vector>
#include "../pmedian/instance.h"
#include "../pmedian/solution.h"

Solution tspmed (const Instance&, const std::vector <int>&, int = 2);

#endif