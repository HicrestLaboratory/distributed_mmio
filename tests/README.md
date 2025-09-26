# Tests for Distributed MMIO

## 1) Setup Test Matrices

Use [MtxMan](https://github.com/ThomasPasquali/MtxMan/tree/main):

```bash
mtxman sync test_matrices.yaml -bmtx -kmtx
```

## 2) Build

```bash
cmake -B build && cmake --build build
```

## 3) Run Tests

Use [SbatchMan](https://sbatchman.readthedocs.io/en/latest/):

```bash
sbatchman init
sbatchman set-cluster-name local
sbatchman configure -f configs.yaml
sbatchman launch -f jobs.yaml
```

## 4) Parse Results

```bash
python3 parse_results.py
```