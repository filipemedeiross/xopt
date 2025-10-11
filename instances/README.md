# OR-Library P-Median Instances

This directory contains benchmark instances for the **p-median problem**, originally provided by the [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/pmedinfo.html) maintained by **J.E. Beasley** at Brunel University.

> These instances are widely used in research on combinatorial optimization and metaheuristic algorithms.

## Contents

- `pmed1` – `pmed40`: classic test instances from the OR-Library
- `pmedbest`: file containing the best-known (or optimal) objective values for each instance

All data files are downloaded directly from the OR-Library website.

## How to Download Automatically

You can automatically download all instances and the `pmedbest.txt` file using the Python script provided in the `scripts` folder.

### Example

```bash
# Run from the project root
python3 -m scripts.1_load_instances
```

This will:

Create the `instances/` directory if it does not exist

Download all pmed1–pmed40 instances

Download the pmedbest.txt file with best-known solutions
