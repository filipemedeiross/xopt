#ifndef TRIE_H
#define TRIE_H

#include <vector>
#include <algorithm>
#include "../pmedian/solution.h"

using namespace std;

struct Node {
    Node* left  = nullptr;
    Node* right = nullptr;

    int visit_count = 0;
};

class SolutionTrie {
    private:
        int n;
        int p;
        Node* root;

        void free_node   (Node*);
        void dfs_collect (const Node*  ,
                          vector <int>&,
                          vector <Solution>&) const;
    public:
        SolutionTrie  (int, int);
        ~SolutionTrie ();

        vector <int> to_binary  (const vector <int >&          ) const;
        int contains            (const vector <int >&          ) const;
        int contains_swap       (const vector <bool>&, int, int) const;
        int contains_and_update (const vector <int>&);

        vector <Solution> get_all_solutions () const;
};

#endif