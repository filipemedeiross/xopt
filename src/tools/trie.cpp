#include "../pmedian/evaluate.h"
#include "trie.h"

SolutionTrie::SolutionTrie (
    int nodes  ,
    int medians
) : n(nodes), p(medians) {
    root = new Node();
}

SolutionTrie::~SolutionTrie () {
    unique_lock <shared_mutex> lock(mutex);

    free_node(root);
}

void SolutionTrie::free_node (Node* node) {
    if (!node)
        return;

    free_node(node->left );
    free_node(node->right);

    delete node;
}

int SolutionTrie::update (const vector <int>& S) {
    vector <int> bits(n, 0);
    for (int v : S)
        bits[v] = 1;

    return update_mask(bits);
}

int SolutionTrie::update_mask (const vector <int>& bits) {
    unique_lock <shared_mutex> lock(mutex);

    Node* node = root;
    for (int bit : bits) {
        Node*& next = bit         ?
                      node->right :
                      node->left;
        if (!next)
            next = new Node();

        node = next;
    }

    return ++node->visit_count;
}

int SolutionTrie::contains (const vector <int>& S) const {
    vector <int> bits(n, 0);
    for (int v : S)
        bits[v] = 1;

    return contains_mask(bits);
}

int SolutionTrie::contains_mask (const vector <int>& bits) const {
    unique_lock <shared_mutex> lock(mutex);

    const Node* node = root;
    for (int bit : bits) {
        node = bit         ?
               node->right :
               node->left;

        if (!node)
            return 0;
    }

    return ++node->visit_count;
}

int SolutionTrie::contains_swap(
    const vector <int>& in_solution,
    int out,
    int in
) const {
    unique_lock <shared_mutex> lock(mutex);

    int i, bit;
    const Node* node = root;

    for (i = 0; i < n; i++) {
        if (i == out)
            bit = 0;
        else if (i == in)
            bit = 1;
        else
            bit = in_solution[i];

        node = bit         ?
               node->right :
               node->left;
        if (!node)
            return 0;
    }

    return ++node->visit_count;
}

vector <Solution> SolutionTrie::get_all_solutions (const Instance& instance) const {
    vector <int     > current;
    vector <Solution> results;

    shared_lock <shared_mutex> lock(mutex);
    dfs_collect (root, current, results, instance);

    return results;
}

void SolutionTrie::dfs_collect (
    const  Node*       node   ,
    vector <int>&      current,
    vector <Solution>& results,
    const  Instance&   instance
) const {
    if (!node)
        return;

    if (node->visit_count && (int) current.size() == n) {
        int i;
        vector <int> facilities;

        facilities.reserve(p);
        for (i = 0; i < n; i++)
            if (current[i] == 1)
                facilities.push_back(i);

        results.push_back({
            evaluate_cost(instance, facilities),
            facilities
        });
    }

    if ((int) current.size() < n) {
        current.push_back(0);
        dfs_collect(node->left , current, results, instance);
        current.pop_back ( );

        current.push_back(1);
        dfs_collect(node->right, current, results, instance);
        current.pop_back ( );
    }
}