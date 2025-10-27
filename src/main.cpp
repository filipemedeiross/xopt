#include <vector>
#include <future>
#include <random>
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

int main (int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <file_path> [restarts]" << endl;
        return 1;
    }

    size_t i, restarts = 5;

    if (argc == 3) {
        int tmp = atoi(argv[2]);

        if (tmp > 0)
            restarts = tmp;
        else
            cerr << "Invalid restarts, using default: " << restarts << endl;
    }

    cout << "Running "
         << restarts << " k-medoids restarts and "
         << restarts << " random restarts..."     << endl;

    Instance instance (argv[1]);

    int p = instance.get_p();
    int n = instance.get_n();
    mt19937 rng(random_device{}());

    vector <TSResult>          solutions;
    vector <future <TSResult>> futures  ;

    solutions.reserve(2 * restarts);
    futures  .reserve(2 * restarts);

    vector <Medoids>      imedoids = kmedoids (instance,
                                               5       ,
                                               restarts);
    vector <vector <int>> irandom (restarts);

    vector <int> idx (n);
    iota (idx.begin(), idx.end(), 0);

    shared_ptr <SolutionTrie> shared_memory = SolutionTrie::get_global_instance (n, p);

    for (i = 0; i < restarts; ++i) {
        shuffle (idx.begin(), idx.end(), rng);
        irandom[i] = vector <int> (idx.begin(), idx.begin() + p);
    }

    for (i = 0; i < restarts; ++i)
        futures.push_back(
            async(
                launch::async,
                [&instance, &irandom, i] () {
                    return tspmed (instance, irandom[i]);
                }
            )
        );
    for (i = 0; i < restarts; i++)
        futures.push_back(
            async (
                launch::async,
                [&instance, &imedoids, i] () {
                    return tspmed (instance, imedoids[i].medoids);
                }
            )
        );

    for (i = 0; i < 2 * restarts; i++)
        solutions.push_back(futures[i].get());

    cout << "Random initials and their results after TS:" << endl;
    for (i = 0; i < restarts; ++i)
        cout << "Restart #" << i + 1                               << ": "
             << evaluate_cost(instance, irandom[i])                << " -> "
             << solutions[i].best.cost                             << " ("
             << solutions[i].long_term->get_all_solutions().size() << " stored solutions)" << endl;

    cout << "Initial medoids and their results after TS:" << endl;
    for (i = restarts; i < 2 * restarts; ++i)
        cout << "Restart #" << i + 1                               << ": "
             << imedoids  [i - restarts].cost                      << " -> "
             << solutions [i           ].best.cost                 << " ("
             << solutions[i].long_term->get_all_solutions().size() << " stored solutions)" << endl;

    sort (
        solutions.begin(),
        solutions.end  (),
        [] (const TSResult& a, const TSResult& b) {
            return a.best.cost < b.best.cost;
        }
    );

    cout << endl;
    cout << "Best solution found:" << endl;
    solutions.front().best.describe();
    cout << "Long-term memory stored "
         << solutions.front().long_term->get_all_solutions().size()
         << " solutions." << endl;

    shuffle (idx.begin(), idx.end(), rng);
    vector <int> initial (idx.begin(), idx.begin() + p);
    TSResult     solution = tspmed (instance, initial);

    cout << endl;
    cout << "Isolated solution:" << endl;
    solution.best.describe();

    cout << endl;
    cout << "Long-term memory collected "
         << solution.long_term->get_all_solutions().size()
         << " unique solutions."          << endl;

    return 0;
}