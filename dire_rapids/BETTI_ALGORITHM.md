# Fast Betti Curve Computation via Incremental Atlas Filtration

## Overview

The `betti_curve.py` module computes filtered Betti curves (`beta_0`, `beta_1`)
for point cloud data using a kNN-based atlas complex. The non-ripser fallback
builds the atlas once, sorts simplex insertion events by filtration value, and
updates Betti numbers incrementally.

## Mathematical Foundation

For a simplicial complex with V vertices, E edges, T triangles, and boundary
operators B1 (V x E) and B2 (E x T):

**beta_0** (connected components) is computed via union-find. As edges enter the
filtration, each edge performs one `union(u, v)`. Path compression and union by
rank make this effectively linear: O(E * alpha(V)).

**beta_1** (independent loops) uses the Hodge decomposition identity:

    beta_1 = E - rank(B1) - rank(B2)

where:
- **rank(B1) = V - beta_0** (standard graph theory: the rank of the incidence
  matrix of a graph equals the number of vertices minus the number of connected
  components)
- **rank(B2)** is maintained incrementally over the field GF(2)

This gives a field-valued Betti computation without floating-point rank
thresholds.

## Implementation Details

### Incremental rank of B2 over GF(2)

B2 is the boundary operator from 2-simplices (triangles) to 1-simplices (edges),
with one column per triangle and one row per edge. Over GF(2), orientation signs
disappear, so each triangle boundary is just a sparse bit vector with exactly
three 1-bits: its three boundary edges.

The implementation stores each column as a Python integer bitset. Adding a
triangle means reducing that bitset against existing pivot columns:

1. Take the highest set bit as the pivot row.
2. If no pivot exists at that row, store this column and increment `rank(B2)`.
3. If a pivot exists, XOR the column with the pivot and continue.
4. If the column reduces to zero, it was dependent and does not change rank.

Each triangle is processed once, when all three of its edges are active. This is
the same sparse Gaussian-elimination idea used by persistent-homology reductions,
but specialized to H1 of the atlas complex.

Concretely, if the active edge list assigns indices `e0 -> 0`, `e1 -> 1`,
`e2 -> 2`, ..., then a triangle with boundary edges `(e2, e5, e9)` is encoded as:

    column = (1 << 2) | (1 << 5) | (1 << 9)

The pivot table maps `highest_set_bit -> reduced_column`. For example, if
`column.bit_length() - 1 == 9` and there is already a stored pivot at row 9, the
new column is replaced by:

    column ^= pivots[9]

This is row cancellation over GF(2): `1 + 1 = 0`, so XOR clears the pivot bit and
may toggle lower bits. The process repeats until either a new pivot row is found
or the column becomes zero. Python integers make this compact: XOR and
`bit_length()` operate on whole machine-word chunks internally rather than on
Python lists of edge indices.

Computing over GF(2) is deliberate: persistent homology is field-valued, GF(2) is
the usual fast default, and it avoids expensive floating-point rank thresholds.
For complexes with torsion, Betti numbers can depend on the chosen field; this
fallback reports field-valued Betti numbers over GF(2). The
`compute_betti_curve_cpu` path remains available as an eigsh-based reference.

### Backend Architecture

Three implementations share the same filtration infrastructure (atlas building,
edge filtering):

| Backend | kNN | Betti computation | Use case |
|---------|-----|-------------------|----------|
| `compute_betti_curve_fast` | sklearn (CPU) | incremental union-find + GF(2) bitset rank | Default for CPU-only |
| `compute_betti_curve_gpu` | cuVS/cuML (GPU) | GPU kNN + incremental union-find/GF(2) bitset rank | Default when GPU available |
| `compute_betti_curve_cpu` | sklearn (CPU) | scipy eigsh (shift-invert) | Reference implementation |

The `compute_betti_curve` selector prefers ripser when installed, then tries the
GPU atlas path, then falls back to the CPU atlas path.

### Performance

The main operations are:

- kNN construction;
- atlas construction, roughly O(N * k^2) local neighbor-neighbor checks;
- one pass over edge events and triangle events;
- sparse GF(2) elimination on triangle boundary bitsets.

On a local Apple Silicon CPU, a noisy 3D circle with `k=15`, `n_steps=50` took
about 0.34s at N=1000 and 1.40s at N=3000. Exact times vary by hardware and
dataset geometry.

### Correctness

The incremental atlas method is exact over GF(2):

- **beta_0**: Union-find counts all connected components exactly, including
  isolated vertices. The eigsh method is capped at k=50 eigenvalues and cannot
  detect more than 50 components.
- **beta_1**: The rank formula is exact over GF(2), and bitset elimination has
  no floating-point tolerance.

Small discrepancies with the eigsh reference can occur because eigsh is
approximate and because eigsh over real coefficients and the atlas fallback over
GF(2) are different coefficient-field protocols.
