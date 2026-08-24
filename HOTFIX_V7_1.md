# V7.1 hotfix

Fixes the Kaggle smoke-test crash:

`TypeError: Object of type PosixPath is not JSON serializable`

Changes:
- JSON writers now serialize `Path` and NumPy scalar/array types safely.
- Restart-selection metadata stores checkpoint paths as strings.
- V7 result defaults now follow the repository convention:
  `results/Adaptive/<instance>/seed_<seed>`.
- Local V7 Gurobi default uses the same `results/Adaptive` root.
- Kaggle output archive now contains `results/Adaptive/...` directly.

No RL objective, reward, shock curriculum, optimizer, architecture, or
evaluation semantics were changed by this hotfix.
