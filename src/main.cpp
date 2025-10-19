#include <vector>
#include <future>
#include <random>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "pmedian/instance.h"
#include "pmedian/solution.h"
#include "pmedian/evaluate.h"
#include "kmedoids/medoids.h"
#include "kmedoids/kmedoids.h"
#include "ts/ts.h"

using namespace std;

int main (int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <file_path> [n_restarts]" << endl;
        return 1;
    }

    size_t i, n_restarts = 5;

    if (argc >= 3) {
        int tmp = atoi(argv[2]);

        if (tmp > 0)
            n_restarts = tmp;
        else
            cerr << "Invalid n_restarts, using default: " << n_restarts << endl;
    }

    cout << "Running "
         << n_restarts << " k-medoids restarts and "
         << n_restarts << " random restarts..."      << endl;

    Instance instance (argv[1]);

    int p = instance.get_p();
    int n = instance.get_n();
    mt19937 rng(random_device{}());

    vector <vector <int>> random_initials (n_restarts);
    vector <Medoids> all_initial = kmedoids (instance, 5, n_restarts);

    for (i = 0; i < n_restarts; ++i) {
        vector <int> idx (n);
        iota   (idx.begin(), idx.end(), 0  );
        shuffle(idx.begin(), idx.end(), rng);

        random_initials[i] = vector <int> (idx.begin(), idx.begin() + p);
    }

    vector <future <Solution>> futures;

    for (i = 0; i < n_restarts; ++i)
        futures.push_back(
            async(
                launch::async,
                [&instance, &random_initials, i] () {
                    return tspmed(instance, random_initials[i]);
                }
            )
        );
    for (i = 0; i < n_restarts; i++)
        futures.push_back(
            async (
                launch::async,
                [&instance, &all_initial, i] () {
                    return tspmed(instance, all_initial[i].medoids);
                }
            )
        );

    vector <Solution> all_solutions;
    for (i = 0; i < 2 * n_restarts; i++)
        all_solutions.push_back(futures[i].get());

    cout << "Initial medoids and their results after TS:" << endl;
    for (i = 0; i < n_restarts; ++i)
        cout << "Restart #" << i + 1  << ": "
             << evaluate(instance, random_initials[i]) << " -> "
             << all_solutions[i].cost                  << endl;

    cout << "Random initials and their results after TS:" << endl;
    for (i = n_restarts; i < 2 * n_restarts; ++i)
        cout << "Restart #" << i + 1  << ": "
             << all_initial  [i - n_restarts].cost << " -> "
             << all_solutions[i             ].cost << endl;

    sort (
        all_solutions.begin(),
        all_solutions.end  (),
        [] (const Solution& a, const Solution& b) {
            return a.cost < b.cost;
        }
    );

    cout << endl;
    cout << "Best solution found:" << endl;
    all_solutions.front().describe();

    return 0;
}