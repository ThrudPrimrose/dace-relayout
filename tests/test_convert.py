# Copyright 2026 ETH Zurich and the relayout authors.
"""The conversion kernels against numpy: element casts bit for bit (ml_dtypes for bfloat16 and fp8),
layouts cell for cell, both directions, built with the host C compiler (``$CC``, default ``cc``); the
device kernels when cupy has a GPU."""

import ctypes
import itertools
import os
import pathlib
import tempfile

import ml_dtypes
import numpy as np
import pytest

from tests.conftest import LAYOUT_CASES, LAYOUT_SHAPE
from relayout.convert import (
    Conversion,
    Group,
    Member,
    compile_host,
    device_source,
    from_layout,
    host_source,
    to_layout,
)
from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, Shuffle, eval_index, run

NARROW = {
    "float16": np.dtype(np.float16),
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float8_e4m3fn": np.dtype(ml_dtypes.float8_e4m3fn),
    "float8_e5m2": np.dtype(ml_dtypes.float8_e5m2),
}


def build(conversions: list[Conversion], tmp: pathlib.Path) -> ctypes.CDLL:
    return ctypes.CDLL(str(compile_host(host_source(conversions), tmp / "convert.so", os.environ.get("CC", "cc"))))


def call(lib: ctypes.CDLL, symbol: str, conversion: Conversion, src: np.ndarray, dst: np.ndarray, **sizes: int) -> None:
    """Kernel ``symbol`` of ``conversion`` on ``src`` -> ``dst``, the shape's symbols bound by name."""
    fn = lib[symbol]
    fn.restype = None
    args = (ctypes.c_int64(sizes[name]) for name in conversion.symbols())
    fn(ctypes.c_void_p(src.ctypes.data), ctypes.c_void_p(dst.ctypes.data), *args)


def probes(source: np.dtype) -> np.ndarray:
    """Random values over many binades, every format's edges, ties, subnormals, signed zero, Inf, NaN."""
    rng = np.random.default_rng(7)
    values = rng.standard_normal(4000) * np.exp2(rng.integers(-30, 20, 4000))
    edges = [0.0, -0.0, np.inf, -np.inf, np.nan, 448.0, 449.0, 464.0, 65504.0, 65519.9, 65520.0, 57344.0, 61440.0]
    edges += [-0.0026474]  # a bfloat16 near-tie ml_dtypes rounds through float32
    for fmt in NARROW.values():
        info = ml_dtypes.finfo(fmt)
        grid = np.array([info.tiny, info.smallest_subnormal, float(info.max)], dtype=np.float64)
        edges += [*grid, *(grid * 1.5), *(grid / 2), *(grid * 0.75)]
        steps = np.arange(-16, 16, dtype=np.float64)
        edges += list((steps + 0.5) * float(info.eps))  # ties between neighbours near 1
    with np.errstate(over="ignore"):  # the overflow edges are the point
        return np.concatenate([values, edges]).astype(source)


@pytest.mark.parametrize("source", ["float64", "float32"])
def test_every_narrow_format_rounds_like_numpy(source: str, tmp_path: pathlib.Path) -> None:
    flat = IndexMap.identity(["n"])
    conversions = [Conversion(f"to_{name}", flat, source, name) for name in NARROW]
    lib = build(conversions, tmp_path)
    values = probes(np.dtype(source))
    for name, dtype in NARROW.items():
        out = np.empty(values.size, dtype=dtype)
        call(lib, f"relayout_in_to_{name}", conversions[0], values, out, n=values.size)
        with np.errstate(over="ignore"):
            expected = values.astype(dtype)
        bits = f"u{dtype.itemsize}"
        both_nan = np.isnan(out.astype(np.float64)) & np.isnan(expected.astype(np.float64))
        same = (out.view(bits) == expected.view(bits)) | both_nan
        bad = np.flatnonzero(~same)
        assert bad.size == 0, f"{source} -> {name}: {values[bad[:5]]} gave {out[bad[:5]]}, numpy {expected[bad[:5]]}"
        back = np.empty(values.size, dtype=np.dtype(source))
        call(lib, f"relayout_out_to_{name}", conversions[0], out, back, n=values.size)
        np.testing.assert_array_equal(back, expected.astype(np.dtype(source)))


def test_a_cast_to_bool_is_not_zero_and_from_int_rounds_like_ml_dtypes(tmp_path: pathlib.Path) -> None:
    """A bool is the byte 1 for every value but zero (0.5, 256 and NaN included), as numpy casts; an int64
    narrows to bfloat16 through float, as ml_dtypes does (2**24 + 2**16 + 1 rounds twice to 2**24)."""
    flat = IndexMap.identity(["n"])
    sources = {
        "float64": np.array([0.0, -0.0, 0.5, -0.25, np.nan, np.inf, 2.0, 1e-300]),
        "int16": np.array([0, 1, -1, 256, -512, 255, 7, 0], dtype=np.int16),
        "float16": np.array([0.0, 0.5, -0.0, np.nan, 6e-8, 1.0], dtype=np.float16),
    }
    conversions = [Conversion(f"bool_{name}", flat, name, "bool") for name in sources]
    conversions.append(Conversion("bf16_int64", flat, "int64", "bfloat16"))
    lib = build(conversions, tmp_path)
    for conversion, (name, values) in zip(conversions, sources.items(), strict=False):
        out = np.full(values.size, 7, dtype=np.uint8)
        call(lib, f"relayout_in_bool_{name}", conversion, values, out, n=values.size)
        np.testing.assert_array_equal(out, values.astype(bool).view(np.uint8), err_msg=name)
    big = np.array([2**24 + 2**16 + 1, -(2**31) - 2**23 - 1, 3, 2**40 + 1], dtype=np.int64)
    out = np.empty(big.size, dtype=NARROW["bfloat16"])
    call(lib, "relayout_in_bf16_int64", conversions[-1], big, out, n=big.size)
    np.testing.assert_array_equal(out.view(np.uint16), big.astype(NARROW["bfloat16"]).view(np.uint16))


def test_the_numpy_form_folds_a_root_over_the_grid() -> None:
    """A shuffle may read root(P0, 2); the grid binds it to a constant, which the numpy form evaluates as the C
    kernel folds it."""
    rotate = Shuffle("A", ("f",), ("(f + root(P0, 2)) % 4",), ("(f - root(P0, 2)) % 4",))
    layout = run([Block("A", "i0", "4", ("o", "f")), rotate], [ArraySpec("A", 1)], (4,), "ops").layout("A", ["8"])
    stored = to_layout(np.arange(8.0), layout, {})
    np.testing.assert_array_equal(stored, [[2, 3, 0, 1], [6, 7, 4, 5]])
    np.testing.assert_array_equal(from_layout(stored, layout, {}), np.arange(8.0))


def layout_for(ops: list, shape: tuple[int, ...]) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", ["N", "M"])


def placed(layout: IndexMap, logical: np.ndarray) -> np.ndarray:
    """The laid-out buffer of ``logical`` computed cell by cell from the layout's own index maps."""
    names = {"N": logical.shape[0], "M": logical.shape[1]}
    extents = tuple(eval_index(e, names, "t") for e in layout.extents())
    out = np.zeros(extents, dtype=logical.dtype)
    position = layout.physical(["x0", "x1"])
    for index in itertools.product(*(range(s) for s in logical.shape)):
        cell = tuple(eval_index(e, {**names, "x0": index[0], "x1": index[1]}, "t") for e in position)
        out[cell] = logical[index]
    return out


def test_a_layout_and_a_cast_are_one_pass_both_ways(tmp_path: pathlib.Path) -> None:
    ops = [Block("A", "i1", "4", ("o", "f")), Pad("A", "i0", "3"), Permute("A", ("o", "i0", "f"))]
    layout = layout_for(ops, (5, 10))
    conversion = Conversion("A", layout, "float64", "float32")
    lib = build([conversion], tmp_path)
    logical = np.random.default_rng(1).standard_normal((5, 10))
    stored = np.full(placed(layout, logical).shape, np.nan, dtype=np.float32)
    call(lib, "relayout_in_A", conversion, logical, stored, N=5, M=10)
    np.testing.assert_array_equal(stored, placed(layout, logical.astype(np.float32)))
    back = np.full_like(logical, np.nan)
    call(lib, "relayout_out_A", conversion, stored, back, N=5, M=10)
    np.testing.assert_array_equal(back, logical.astype(np.float32).astype(np.float64))


def test_a_shuffle_keeps_python_floor_semantics_and_its_buffer(tmp_path: pathlib.Path) -> None:
    """A rotation whose inverse subtracts (C's ``%`` would give -1) packs like the layout's own maps, and an
    unchecked swizzle over a dimension of odd extent leaves the cell it maps out of the array dead."""
    rotate = Shuffle("A", ("f",), ("(f + 1) % 4",), ("(f - 1) % 4",))
    swizzle = Shuffle("A", ("i0",), ("i0 ^ 1",), ("i0 ^ 1",))
    logical = np.random.default_rng(2).standard_normal((5, 8))
    layout = layout_for([Block("A", "i1", "4", ("o", "f")), rotate], (5, 8))
    conversion = Conversion("A", layout, "float64", "float64")
    swizzled = Conversion("B", layout_for([swizzle], (5, 8)), "float64", "float64")
    lib = build([conversion, swizzled], tmp_path)
    stored = np.full(placed(layout, logical).shape, np.nan)
    call(lib, "relayout_in_A", conversion, logical, stored, N=5, M=8)
    np.testing.assert_array_equal(stored, placed(layout, logical))
    stored = np.full((5, 8), np.nan)
    call(lib, "relayout_in_B", swizzled, logical, stored, N=5, M=8)
    np.testing.assert_array_equal(stored[:4], logical[[1, 0, 3, 2]])
    np.testing.assert_array_equal(stored[4], 0.0)  # row 4 holds row 5, which the array has not


@pytest.mark.parametrize("case", LAYOUT_CASES)
def test_every_layout_case_packs_and_unpacks_in_c(case: str, tmp_path: pathlib.Path) -> None:
    """The C kernels pack each layout case cell for cell as its index map places it, dead cells zero, and
    unpack it back to the logical array."""
    layout = layout_for(LAYOUT_CASES[case], LAYOUT_SHAPE)
    conversion = Conversion("A", layout, "float64", "float64")
    lib = build([conversion], tmp_path)
    logical = np.random.default_rng(3).standard_normal(LAYOUT_SHAPE)
    sizes = dict(zip(("N", "M"), LAYOUT_SHAPE, strict=True))
    stored = np.full(placed(layout, logical).shape, np.nan)
    call(lib, "relayout_in_A", conversion, logical, stored, **sizes)
    np.testing.assert_array_equal(stored, placed(layout, logical))
    back = np.full_like(logical, np.nan)
    call(lib, "relayout_out_A", conversion, stored, back, **sizes)
    np.testing.assert_array_equal(back, logical)


def test_complex_arrays_cast_their_parts(tmp_path: pathlib.Path) -> None:
    conversion = Conversion("z", layout_for([Permute("A", ("i1", "i0"))], (3, 4)), "complex128", "complex64")
    lib = build([conversion], tmp_path)
    z = np.random.default_rng(2).standard_normal((3, 4)) + 1j * np.random.default_rng(3).standard_normal((3, 4))
    stored = np.empty((4, 3), dtype=np.complex64)
    call(lib, "relayout_in_z", conversion, z, stored, N=3, M=4)
    np.testing.assert_array_equal(stored, z.T.astype(np.complex64))


def group_call(lib: ctypes.CDLL, symbol: str, buffers: list[np.ndarray], group: Group, **sizes: int) -> None:
    fn = getattr(lib, symbol)
    fn(*(ctypes.c_void_p(b.ctypes.data) for b in buffers), *(ctypes.c_int64(sizes[n]) for n in group.symbols()))


def test_a_group_zips_and_unzips_stacked_struct_and_complex(tmp_path: pathlib.Path) -> None:
    """Two parts stacked first (and the whole transposed), a float32 and a float64 part in one struct, and a
    complex array unzipped into its real and imaginary parts: zip then unzip gives the parts back."""
    rng = np.random.default_rng(11)
    flat = IndexMap.identity(["d0", "d1"])
    x, y = rng.random((3, 5)), rng.random((3, 5))
    whole = run([Permute("A", ("i2", "i1", "i0"))], [ArraySpec("A", 3)], (), "ops").layout("A", ["2", "d0", "d1"])
    stacked = Group("s", whole, (Member("x", flat, "float64", "float64"), Member("y", flat, "float64", "float64")), 0)
    a, b = rng.random((3, 5)).astype(np.float32), rng.random((3, 5))
    struct = Group(
        "t",
        flat,
        (Member("a", flat, "float32", "float32", "a"), Member("b", flat, "float64", "float64", "b")),
        None,
        "relayout_struct_t",
    )
    z = rng.random((3, 5)) + 1j * rng.random((3, 5))
    parts = (Member("re", flat, "float64", "float64", "re"), Member("im", flat, "float64", "float64", "im"))
    complex_group = Group("c", flat, parts, None, "relayout_c128")
    lib = ctypes.CDLL(
        str(
            compile_host(
                host_source([], groups=[stacked, struct, complex_group]),
                tmp_path / "groups.so",
                os.environ.get("CC", "cc"),
            )
        )
    )
    s = np.zeros((5, 3, 2))
    group_call(lib, "relayout_zip_s", [x, y, s], stacked, d0=3, d1=5)
    np.testing.assert_array_equal(s, np.stack([x, y]).transpose(2, 1, 0))
    back = [np.zeros((3, 5)), np.zeros((3, 5))]
    group_call(lib, "relayout_unzip_s", [s, *back], stacked, d0=3, d1=5)
    np.testing.assert_array_equal(back[0], x)
    np.testing.assert_array_equal(back[1], y)
    t = np.zeros((3, 5), np.dtype([("a", np.float32), ("b", np.float64)], align=True))
    group_call(lib, "relayout_zip_t", [a, b, t], struct, d0=3, d1=5)
    np.testing.assert_array_equal(t["a"], a)
    np.testing.assert_array_equal(t["b"], b)
    re, im = np.zeros((3, 5)), np.zeros((3, 5))
    group_call(lib, "relayout_unzip_c", [z, re, im], complex_group, d0=3, d1=5)
    np.testing.assert_array_equal(re + 1j * im, z)


def test_a_group_keeps_an_unchecked_renumbering_inside_its_buffers(tmp_path: pathlib.Path) -> None:
    """Two ``[d0]`` parts stacked first, the whole's cell index renumbered by ``i1 + 1`` over the symbolic
    ``d0`` (not provable, so unchecked): the zip writes only the cells that land inside the whole (the last
    cell of each part has none), never past the buffer, and the unzip reads only those."""
    shifted = Shuffle("A", ("i1",), ("i1 + 1",), ("i1 - 1",))
    whole = run([shifted], [ArraySpec("A", 2)], (), "ops").layout("A", ["2", "d0"])
    assert not whole.renumbers[0].checked
    flat = IndexMap.identity(["d0"])
    group = Group("s", whole, (Member("x", flat, "float64", "float64"), Member("y", flat, "float64", "float64")), 0)
    lib = ctypes.CDLL(str(compile_host(host_source([], groups=[group]), tmp_path / "g.so", os.environ.get("CC", "cc"))))
    x, y = np.arange(1.0, 6.0), np.arange(11.0, 16.0)
    guarded = np.full(2 * 5 + 4, -1.0)  # the whole's 10 cells, then 4 guard cells
    group_call(lib, "relayout_zip_s", [x, y, guarded], group, d0=5)
    expected = np.full(14, -1.0)
    expected[:10].reshape(2, 5)[:, 1:] = np.stack([x, y])[:, :4]
    np.testing.assert_array_equal(guarded, expected)
    back = [np.zeros(5), np.zeros(5)]
    group_call(lib, "relayout_unzip_s", [guarded, *back], group, d0=5)
    np.testing.assert_array_equal(np.stack(back), np.concatenate([np.stack([x, y])[:, :4], np.zeros((2, 1))], axis=1))


def test_a_group_of_0d_parts_builds_and_zips(tmp_path: pathlib.Path) -> None:
    """A struct of two 0-d parts (scalars held in arrays) is one cell, with no loop to parallelize: its kernels
    build and move the cell both ways."""
    point = IndexMap.identity([])
    parts = (Member("a", point, "float32", "float32", "a"), Member("b", point, "float64", "float64", "b"))
    group = Group("t", point, parts, None, "relayout_struct_t")
    lib = ctypes.CDLL(str(compile_host(host_source([], groups=[group]), tmp_path / "g.so", os.environ.get("CC", "cc"))))
    a, b = np.array(1.5, np.float32), np.array(2.25)
    whole = np.zeros((), np.dtype([("a", np.float32), ("b", np.float64)], align=True))
    group_call(lib, "relayout_zip_t", [a, b, whole], group)
    assert (whole["a"], whole["b"]) == (1.5, 2.25)
    back = [np.zeros((), np.float32), np.zeros(())]
    group_call(lib, "relayout_unzip_t", [whole, *back], group)
    assert (back[0], back[1]) == (1.5, 2.25)


def test_the_device_kernels_match_the_host(tmp_path: pathlib.Path) -> None:
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no GPU")
    ops = [Block("A", "i1", "4", ("o", "f")), Pad("A", "i0", "3"), Permute("A", ("o", "i0", "f"))]
    layout = layout_for(ops, (5, 10))
    module = cp.RawModule(code=device_source([Conversion("A", layout, "float64", "bfloat16")]), options=("-std=c++17",))
    logical = np.random.default_rng(4).standard_normal((5, 10))
    expected = placed(layout, logical.astype(ml_dtypes.bfloat16)).view(np.uint16)
    stored = cp.zeros(expected.shape, dtype=cp.uint16)
    module.get_function("relayout_in_A")((32,), (128,), (cp.asarray(logical), stored, np.int64(10), np.int64(5)))
    np.testing.assert_array_equal(cp.asnumpy(stored), expected)


if __name__ == "__main__":
    test_a_group_zips_and_unzips_stacked_struct_and_complex(pathlib.Path(tempfile.mkdtemp()))
    test_a_group_keeps_an_unchecked_renumbering_inside_its_buffers(pathlib.Path(tempfile.mkdtemp()))
    test_a_group_of_0d_parts_builds_and_zips(pathlib.Path(tempfile.mkdtemp()))
    for source in ("float64", "float32"):
        with tempfile.TemporaryDirectory() as tmp:
            test_every_narrow_format_rounds_like_numpy(source, pathlib.Path(tmp))
    test_the_numpy_form_folds_a_root_over_the_grid()
    with tempfile.TemporaryDirectory() as tmp:
        test_a_cast_to_bool_is_not_zero_and_from_int_rounds_like_ml_dtypes(pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_a_layout_and_a_cast_are_one_pass_both_ways(pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_a_shuffle_keeps_python_floor_semantics_and_its_buffer(pathlib.Path(tmp))
    for name in LAYOUT_CASES:
        with tempfile.TemporaryDirectory() as tmp:
            test_every_layout_case_packs_and_unpacks_in_c(name, pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_complex_arrays_cast_their_parts(pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            test_the_device_kernels_match_the_host(pathlib.Path(tmp))
        except pytest.skip.Exception as skipped:
            print(f"skipped: {skipped}")
