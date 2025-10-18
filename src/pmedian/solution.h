#ifndef SOLUTION_H
#define SOLUTION_H

#include <iostream>
#include <vector>

using namespace std;

struct Solution {
    double cost;
    vector <int> facilities;

    void describe () const {
        cout << "Best cost: " << cost << endl;

        cout << "Facilities: ";
        for (int v : facilities)
            cout << v + 1 << " ";

        cout << endl;
    }
};

#endif