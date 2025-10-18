#ifndef KMEDOIDS_H
#define KMEDOIDS_H

#include <vector>
#include "../pmedian/instance.h"

std::vector <int> kmedoids (const Instance&, int = 5, int = 3);

#endif