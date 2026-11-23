# Changelog

## 0.1.1 (2026-11-23)

- test_search_assign: C reads B transposed and reversed (one affine access), which no interchange aligns with B's writer; reading B both ways gave a Min() index and dropped A from the middle kernel (f51eea7)
- DaCe extended 1.0.0: subsets.union takes the SDFG's facts; the assignment test's C reads B both ways, since canonicalize now interchanges a lone transposed reader into its writer's map and fuses the two (7feae24)
- Depend on dace-extended (DaCe extended on PyPI, import dace) in place of plain dace; upstream dace is never installed (0dd868e)
- Wheel from a checkout: exclude the corpus symlink so relayout/corpus is added once (uv's git builds failed on a duplicate __init__.py) (0b0e7c7)
- release_patch.sh: no commits since the last tag is nothing to release, not an error (8e110f3)
- helpers/release_patch.sh: patch release vX.Y.Z -> vX.Y.(Z+1) with tag, GitHub release and PyPI upload (b2d2744)

## 0.1.0 (2026-11-23)

The first release, published on PyPI as `dace-relayout` (import `relayout`).

- **Requests.** One request schema (`relayout.requests`, `relayout.ops`) says how a kernel's arrays are laid out,
  cast and distributed: permutations, blockings, casts and MPI / NCCL / RCCL distributions, checked before any
  rewrite.
- **Rewrites.** The same request lowers a numpy / mpi4py program or a DaCe SDFG (`relayout.passes`) and generates the
  kernels that convert the data between layouts (`relayout.convert`).
- **Corpus.** Reference kernels from stream, transpose and sparse kernels to ICON, CLOUDSC and OMEN (`relayout.corpus`),
  with an OMEN quickstart (`examples/omen_quickstart.py`).
- **Python.** 3.12 and later, no upper bound. DaCe follows the `extended` branch, installed from GitHub (see the
  README); the `fortran` and `fparena` extras need a checkout.
- **License.** GPL-3.0-or-later.
