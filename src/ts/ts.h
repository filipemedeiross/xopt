#ifndef TS_H
#define TS_H

#include <memory>
#include <vector>
#include "../pmedian/instance.h"
#include "../pmedian/solution.h"
#include "../tools/trie.h"

struct TSResult {
    Solution                  best;
    unique_ptr <SolutionTrie> long_term;

    TSResult (int n, int p) :
        best      (),
        long_term (make_unique <SolutionTrie> (n, p))
    {}
};

TSResult tspmed (const Instance&, const std::vector <int>&, int = 2);

#endif