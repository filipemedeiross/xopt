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
         << restarts << " random restarts..." << endl;

    Instance instance (argv[1]);

    int p = instance.get_p();
    int n = instance.get_n();
    mt19937 rng(random_device{}());

    vector <Solution>          solutions;
    vector <future <Solution>> futures  ;

    vector <Medoids>      imedoids = kmedoids (instance,
                                               5       ,
                                               restarts);
    vector <vector <int>> irandom (restarts);

    vector <int> idx (n);
    iota (idx.begin(), idx.end(), 0);

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
        cout << "Restart #" << i + 1  << ": "
             << evaluate_cost(instance, irandom[i]) << " -> "
             << solutions[i].cost                   << endl;

    cout << "Initial medoids and their results after TS:" << endl;
    for (i = restarts; i < 2 * restarts; ++i)
        cout << "Restart #" << i + 1  << ": "
             << imedoids  [i - restarts].cost << " -> "
             << solutions [i           ].cost << endl;

    sort (
        solutions.begin(),
        solutions.end  (),
        [] (const Solution& a, const Solution& b) {
            return a.cost < b.cost;
        }
    );

    cout << endl;
    cout << "Best solution found:" << endl;
    solutions.front().describe();

    shuffle (idx.begin(), idx.end(), rng);

    vector <int> initial (idx.begin(), idx.begin() + p);
    Solution     solution = tspmed (instance, initial);

    cout << endl;
    cout << "Isolated solution:" << endl;
    solution.describe();

    return 0;
}