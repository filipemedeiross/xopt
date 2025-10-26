#include "trie.h"

SolutionTrie::SolutionTrie (
    int nodes  ,
    int medians
) : n(nodes), p(medians) {
    root = new Node();
}

SolutionTrie::~SolutionTrie () {
    free_node(root);
}

void SolutionTrie::free_node (Node* node) {
    if (!node)
        return;

    free_node(node->left);
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

    const Node* node = root;
    for (int bit : bits) {
        node = (bit == 0   ?
                node->left :
                node->right);

        if (!node)
            return false;
    }

    return node->visit_count;
}

vector <Solution> SolutionTrie::get_all_solutions () const {
    vector <int     > current;
    vector <Solution> results;
    dfs_collect(root, current, results);

    return results;
}

void SolutionTrie::dfs_collect (
    const Node*   node   ,
    vector <int>& current,
    vector <Solution>& results
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
            0.0,
            facilities
        });
    }

    if ((int) current.size() < n) {
        current.push_back(0);
        dfs_collect(node->left , current, results);
        current.pop_back ( );

        current.push_back(1);
        dfs_collect(node->right, current, results);
        current.pop_back ( );
    }
}