#include <vector>
#include <future>
#include <random>
#include <memory>
#include <cstdlib>
#include <numeric>
#include <iostream>
#include <algorithm>

#include "pmedian/instance.h"
#include "pmedian/solution.h"
#include "pmedian/evaluate.h"
#include "kmedoids/medoids.h"
#include "kmedoids/kmedoids.h"
#include "ts/ts.h"

using namespace std;

namespace {
    vector <int> random_initial (mt19937& rng, vector <int>& indices, int p) {
        shuffle(indices.begin(), indices.end(), rng);
        return vector <int> (indices.begin(), indices.begin() + p);
    }
}

int main (int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <file_path> [restarts]" << endl;
        return 1;
    }

    size_t i, total_runs, restarts = 5;

    if (argc == 3) {
        int tmp = atoi(argv[2]);

        if (tmp > 0)
            restarts = tmp;
        else
            cerr << "Invalid restarts, using default: " << restarts << endl;
    }
    total_runs = 2 * restarts;

    cout << "Running "
         << restarts << " k-medoids restarts and "
         << restarts << " random restarts..."     << endl;

    Instance instance (argv[1]);

    int p = instance.get_p();
    int n = instance.get_n();
    mt19937 rng(random_device{}());

    vector <int> indices(n);
    iota (indices.begin(), indices.end(), 0);

    vector     <TSResult>         solutions;
    vector     <future<TSResult>> futures  ;
    shared_ptr <SolutionTrie>     memory = make_shared <SolutionTrie> (n, p);

    solutions.reserve(total_runs);
    futures  .reserve(total_runs);

    vector <Medoids>      imedoids = kmedoids(instance,
                                              5       ,
                                              restarts);
    vector <vector <int>> irandom(restarts);

    for (i = 0; i < restarts; ++i)
        irandom[i] = random_initial (rng, indices, p);

    for (i = 0; i < restarts; ++i) {
        futures.push_back(
            async(
                launch::async,
                [memory, &instance, &irandom, i] () {
                    return tspmed (instance, irandom[i], 2, memory);
                }
            )
        );
    }
    for (i = 0; i < restarts; i++) {
        futures.push_back(
            async (
                launch::async,
                [memory, &instance, &imedoids, i] () {
                    return tspmed (instance, imedoids[i].medoids, 2, memory);
                }
            )
        );
    }

    for (i = 0; i < total_runs; i++)
        solutions.push_back(futures[i].get());

    cout << "Random initials and their results after TS:" << endl;
    for (i = 0; i < restarts; ++i)
        cout << "Restart #" << i + 1                                       << ": "
             << evaluate_cost(instance, irandom[i])                        << " -> "
             << solutions[i].best.cost                                     << " ("
             << solutions[i].long_term->get_all_solutions(instance).size() << " stored solutions)" << endl;

    cout << "Initial medoids and their results after TS:" << endl;
    for (i = restarts; i < total_runs; ++i)
        cout << "Restart #" << i + 1                                       << ": "
             << imedoids  [i - restarts].cost                              << " -> "
             << solutions [i           ].best.cost                         << " ("
             << solutions[i].long_term->get_all_solutions(instance).size() << " stored solutions)" << endl;

    sort (
        solutions.begin(),
        solutions.end  (),
        [] (const TSResult& a, const TSResult& b) {
            return a.best.cost < b.best.cost;
        }
    );

    TSResult best_sol = solutions.front();

    cout << endl;
    cout << "Best solution found:" << endl;

    best_sol.best.describe();
    cout << "Long-term memory stored "
         << best_sol.long_term->get_all_solutions(instance).size()
         << " solutions." << endl;

    vector <int> initial  = random_initial (rng, indices, p);
    TSResult     solution = tspmed (instance, initial);

    cout << endl;
    cout << "Isolated solution:" << endl;

    solution.best.describe();
    cout << endl;
    cout << "Long-term memory collected "
         << solution.long_term->get_all_solutions(instance).size()
         << " unique solutions."          << endl;

    return 0;
}