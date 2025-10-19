#ifndef KMEDOIDS_H
#define KMEDOIDS_H

#include <vector>
#include "../pmedian/instance.h"
#include "medoids.h"

std::vector <Medoids> kmedoids (const Instance&, int = 10, int = 5);

#endif