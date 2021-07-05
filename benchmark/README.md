# Benchmarking of NetworkDynamics

The ND benchmarks are defined in [benchmarks.jl](benchmarks.jl).

All benchmarks belong to a central benchmark suite `SUITE`. A benchmark
suit is basically a dictionary of defined benchmarks. After a run
of the benchmarks one can access the individual trials via the same syntax.

The benchmarks will be run using `PkgBenchmark.jl`. Make sure that you
have this pkg in your global environment so it is loadable! Otherwise we'd
have to add this to the `NetworkDynamics.jl` deps...

The easiest way to run the benchmarks is
```
$ cd NetworkDynamics/benchmarks
$ ./run_benchmarks.jl
```
which will run the benchmark defined in the current directory

- for the currently checked out version of ND
- for main

and export the results to `target.md`, `baseline.md` and `judgment.md`.

Arguments:
- `-t, --target`: Specify branch, commit id, tag, ... for target benchmark. If `directory` use the current state of the directory. Default: `directory`
- `-b, --baseline`: Same for baseline benchmark. Default: `main`. If `none` or `nothing` skip the baseline benckmark and coparison step
- `--command`: Julia command to use for benchmarks, defaults to `julia`
- `--tcommand`: Julia command for target, if unset use `--command`
- `--bcommand`: Julia command for baseline, if unset use `--command`
- `--threads`: set thread number env variable, defaults to `4`
- `-v, --verbose`: show the current benchmark during run
- `-p, --prefix`: prefix for filenames, defaults to timestamp
- `--retune`: Force [retuning](https://juliaci.github.io/BenchmarkTools.jl/dev/manual/#Caching-Parameters) before the first benchmark (second benchmark will use the tune file).
- `--exportraw`: export raw data of trials, i.e. to perform your own, additional analysis

See the [`PkgBenchmark.jl` docs](https://juliaci.github.io/PkgBenchmark.jl/stable/) for more details.

