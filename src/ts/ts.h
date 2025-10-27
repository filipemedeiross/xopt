#ifndef TS_H
#define TS_H

#include <memory>
#include <vector>
#include "../pmedian/instance.h"
#include "../pmedian/solution.h"
#include "../tools/trie.h"

using namespace std;

struct TSResult {
    Solution   best {0.0, {}};
    shared_ptr <SolutionTrie> long_term;

    TSResult (shared_ptr <SolutionTrie> solution) : long_term(solution) {}
};

TSResult tspmed (const Instance&, const vector <int>&, int = 2);

#endif