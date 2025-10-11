#include <iostream>
#include <fstream>
#include <iomanip>
#include "instance.h"

using namespace std;

const int Instance::INF = 1e9;

Instance::Instance () : n(0), p(0) {}

Instance::Instance (const string& filename) {
    if (!load(filename))
        cerr << "Error loading file: " << filename << endl;

    floyd_warshall();
}

bool Instance::load (const string& filename) {
    int m;
    int i, e;
    int u, v, c;

    ifstream file(filename);
    if (!file.is_open())
        return false;

    file >> n >> m >> p;

    dist.assign(n, vector<int> (n, INF));
    for (i = 0; i < n; i++)
        dist[i][i] = 0;

    for (e = 0; e < m; e++) {
        file >> u >> v >> c;

        u--; v--;
        dist[u][v] = c;
        dist[v][u] = c;
    }

    file.close();

    return true;
}

void Instance::floyd_warshall () {
    int k, i, j;

    for (k = 0; k < n; k++)
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
}

void Instance::describe () const {
    int i, j;

    cout << "Number of vertices: " << n << endl;
    cout << "Number of medians: "  << p << endl;
    cout << "Shortest distance matrix:" << endl;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            if (dist[i][j] >= INF / 2)
                cout << setw(4) << "INF";
            else
                cout << setw(4) << dist[i][j];

        cout << endl;
    }
}