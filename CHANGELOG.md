# Changelog

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
