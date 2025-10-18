#include <iostream>
#include "pmedian/instance.h"
#include "pmedian/solution.h"
#include "ts/ts.h"

using namespace std;

int main (int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Use: " << argv[0] << " <file_path>" << endl;
        return 1;
    }

    Instance instance (argv[1]);
    Solution solution = tspmed (instance);

    solution.describe();

    return 0;
}