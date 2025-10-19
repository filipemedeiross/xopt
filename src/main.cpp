#include <vector>
#include <future>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "pmedian/instance.h"
#include "pmedian/solution.h"
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

    cout << "Running " << n_restarts << " restarts..." << endl;

    Instance instance (argv[1]);

    vector <Medoids>  all_initial = kmedoids (instance, 5, n_restarts);
    vector <Solution> all_solutions;
    vector <future <Solution>> futures;

    for (i = 0; i < n_restarts; i++)
        futures.push_back(
            async (
                launch::async,
                [&instance, &all_initial, i] () {
                    return tspmed(instance, all_initial[i].medoids);
                }
            )
        );

    for (i = 0; i < n_restarts; i++)
        all_solutions.push_back(futures[i].get());

    cout << "Initial medoids and their results after TS:" << endl;
    for (i = 0; i < n_restarts; ++i)
        cout << "Restart #" << i + 1  << ": "
             << all_initial  [i].cost << " -> "
             << all_solutions[i].cost << endl;

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