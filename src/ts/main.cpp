#include <iostream>
#include "instance.h"

using namespace std;

int main (int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Use: " << argv[0] << " <file_path>" << endl;
        return 1;
    }

    Instance instance(argv[1]);
    instance.describe();

    return 0;
}