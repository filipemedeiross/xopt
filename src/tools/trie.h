#ifndef TRIE_H
#define TRIE_H

#include <mutex>
#include <memory>
#include <vector>
#include <algorithm>
#include <shared_mutex>
#include "../pmedian/instance.h"
#include "../pmedian/solution.h"

using namespace std;

struct Node {
    Node* left  = nullptr;
    Node* right = nullptr;

    mutable int visit_count = 0;
};

class SolutionTrie {
    private:
        int   n;
        int   p;
        Node* root;

        mutable shared_mutex mutex;

        void free_node   (Node*);
        void dfs_collect (const Node*       ,
                          vector <int>&     ,
                          vector <Solution>&,
                          const Instance&   ) const;
    public:
        SolutionTrie  (int, int);
        ~SolutionTrie ();

        int update   (const vector <int>&);
        int contains (const vector <int>&) const;
        int contains_swap (const vector <bool>&, int, int) const;

        vector <int>      to_binary         (const vector <int>&) const;
        vector <Solution> get_all_solutions (const Instance&    ) const;
};

#endif