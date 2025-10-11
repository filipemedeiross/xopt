#ifndef INSTANCE_H
#define INSTANCE_H

#include <vector>
#include <string>

using namespace std;

class Instance {
    int n;
    int p;
    vector <vector <int>> dist;

    public:
        static const int INF;

        Instance ();
        Instance (const string& filename);

        bool load (const string& filename);
        void floyd_warshall ();

        void describe () const;

        int get_n () const { return n; }
        int get_p () const { return p; }
        int operator () (int i, int j) const { return dist[i][j]; }
};

#endif