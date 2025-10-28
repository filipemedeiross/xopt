#include "../pmedian/evaluate.h"
#include "trie.h"

mutex                     SolutionTrie::global_mutex;
shared_ptr <SolutionTrie> SolutionTrie::global_instance = nullptr;

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

shared_ptr <SolutionTrie> SolutionTrie::get_global_instance (
    int nodes  ,
    int medians
) {
    lock_guard <std::mutex> guard(global_mutex);

    if (!global_instance)
        global_instance = make_shared <SolutionTrie> (nodes, medians);

    return global_instance;
}

void SolutionTrie::free_node (Node* node) {
    if (!node)
        return;

    free_node(node->left );
    free_node(node->right);

    delete node;
}

vector <int> SolutionTrie::to_binary (const vector <int>& S) const {
    vector <int> bits(n, 0);
    for (int v : S)
        bits[v] = 1;

    return bits;
}

int SolutionTrie::contains_and_update (const vector <int>& S) {
    vector <int> bits = to_binary(S);

    unique_lock <shared_mutex> lock(mutex);

    Node* node = root;
    for (int bit : bits) {
        Node*& next = (bit == 0   ?
                       node->left :
                       node->right);
        if (!next)
            next = new Node();

        node = next;
    }

    return ++node->visit_count;
}

int SolutionTrie::contains (const vector <int>& S) const {
    vector <int> bits = to_binary(S);

    unique_lock <shared_mutex> lock(mutex);
    const Node* node = root;
    for (int bit : bits) {
        node = (bit == 0   ?
                node->left :
                node->right);

        if (!node)
            return 0;
    }

    return ++node->visit_count;
}

int SolutionTrie::contains_swap(
    const vector <bool>& in_solution,
    int out,
    int in
) const {
    int i, bit;
    const Node* node = root;

    unique_lock <shared_mutex> lock(mutex);
    for (i = 0; i < n; i++) {
        if      (i == out)
            bit = 0;
        else if (i == in )
            bit = 1;
        else
            bit = in_solution[i] ? 1 : 0;

        node = (bit == 0   ?
                node->left :
                node->right);
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