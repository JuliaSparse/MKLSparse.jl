# Public inspector-executor / two-stage SpBLAS API.
#
# Exposes the MKL Sparse BLAS inspector-executor entry points
# (mkl_sparse_optimize, mkl_sparse_set_*_hint, mkl_sparse_set_memory_hint)
# and the separable two-stage SpGEMM (mkl_sparse_sp2m with SPARSE_STAGE_NNZ_COUNT
# / SPARSE_STAGE_FINALIZE_MULT) so that callers can analyze a sparsity pattern
# once and execute the numeric phase many times.
#
# See:
# https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/inspector-executor-sparse-blas-execution-routines.html

# ---------- mkl_sparse_optimize ---------------------------------------------

"""
    optimize!(A::MKLSparseMatrix) -> MKLSparseMatrix

Run `mkl_sparse_optimize` on `A`. After this call MKL is allowed to compute and
cache an internal representation of `A` tailored to the operations announced via
`set_mv_hint!`, `set_mm_hint!`, `set_dotmv_hint!`, `set_memory_hint!`. Any further
`mv!` / `mm!` / `sp2m!` against `A` may use that cached representation.

Idempotent — calling `optimize!` repeatedly is safe.
"""
function optimize!(A::MKLSparseMatrix{S}) where S
    res = mkl_call(Val{:mkl_sparse_optimizeI}(), S, A.handle, log=Val{false}())
    check_status(res)
    return A
end

# ---------- hints -----------------------------------------------------------

"""
    set_mv_hint!(A::MKLSparseMatrix; operation='N', descr=matrix_descr('G','F','N'),
                                     expected_calls=1) -> MKLSparseMatrix

Tell MKL to expect `expected_calls` invocations of `op(A) * x` (matrix–vector).
A subsequent `optimize!(A)` may then specialize storage for SpMV.

`operation` accepts the chars `'N'`, `'T'`, `'C'`, or a `sparse_operation_t`.
"""
function set_mv_hint!(A::MKLSparseMatrix{S};
                      operation = 'N',
                      descr::matrix_descr = matrix_descr('G', 'F', 'N'),
                      expected_calls::Integer = 1
) where S
    op = convert(sparse_operation_t, operation)
    res = mkl_call(Val{:mkl_sparse_set_mv_hintI}(), S,
                   A.handle, op, descr, expected_calls, log=Val{false}())
    check_status(res)
    return A
end

"""
    set_mm_hint!(A::MKLSparseMatrix; operation='N', descr=matrix_descr('G','F','N'),
                                     layout='C', dense_cols=1, expected_calls=1) -> MKLSparseMatrix

Tell MKL to expect `expected_calls` invocations of `op(A) * X` where `X` is a
dense matrix with `dense_cols` columns laid out per `layout` (`'C'` column-major,
`'R'` row-major).
"""
function set_mm_hint!(A::MKLSparseMatrix{S};
                      operation = 'N',
                      descr::matrix_descr = matrix_descr('G', 'F', 'N'),
                      layout = 'C',
                      dense_cols::Integer = 1,
                      expected_calls::Integer = 1
) where S
    op = convert(sparse_operation_t, operation)
    lay = convert(sparse_layout_t, layout)
    res = mkl_call(Val{:mkl_sparse_set_mm_hintI}(), S,
                   A.handle, op, descr, lay, dense_cols, expected_calls,
                   log=Val{false}())
    check_status(res)
    return A
end

"""
    set_dotmv_hint!(A::MKLSparseMatrix; operation='N', descr=matrix_descr('G','F','N'),
                                        expected_calls=1) -> MKLSparseMatrix

Tell MKL to expect `expected_calls` invocations of the fused `mv + dot` routine.
"""
function set_dotmv_hint!(A::MKLSparseMatrix{S};
                         operation = 'N',
                         descr::matrix_descr = matrix_descr('G', 'F', 'N'),
                         expected_calls::Integer = 1
) where S
    op = convert(sparse_operation_t, operation)
    res = mkl_call(Val{:mkl_sparse_set_dotmv_hintI}(), S,
                   A.handle, op, descr, expected_calls, log=Val{false}())
    check_status(res)
    return A
end

"""
    set_memory_hint!(A::MKLSparseMatrix; policy=:aggressive) -> MKLSparseMatrix

Tell MKL whether it is allowed to allocate auxiliary memory for `A`. `policy` is
either `:aggressive` (`SPARSE_MEMORY_AGGRESSIVE`) or `:none` (`SPARSE_MEMORY_NONE`),
or a literal `sparse_memory_usage_t` enum value.
"""
function set_memory_hint!(A::MKLSparseMatrix{S};
                          policy = :aggressive
) where S
    pol = policy isa sparse_memory_usage_t ? policy :
          policy === :aggressive ? SPARSE_MEMORY_AGGRESSIVE :
          policy === :none       ? SPARSE_MEMORY_NONE       :
          throw(ArgumentError(lazy"Unknown memory policy $policy"))
    res = mkl_call(Val{:mkl_sparse_set_memory_hintI}(), S,
                   A.handle, pol, log=Val{false}())
    check_status(res)
    return A
end

# ---------- separable two-stage SpGEMM --------------------------------------
#
# C = op(A) * op(B), with the sparsity-pattern phase (NNZ_COUNT) and the
# numeric phase (FINALIZE_MULT) callable separately so each can be timed and
# instrumented (e.g., with PAPI counters) on its own.

"""
    sp2m_nnz_count(transA, A::MKLSparseMatrix{S}, descrA,
                   transB, B::MKLSparseMatrix{S}, descrB) -> hC::MKLSparseMatrix{S}

Stage 1 of the two-stage `sp2m`: compute the sparsity pattern (the row/col
pointers and minor indices) of `C = op(A) * op(B)` and return a new MKL handle
`hC` that owns that pattern. The values are **not** computed yet — call
`sp2m_finalize_mult!(hC, ...)` next.

The returned handle must be `destroy`-ed by the caller.

`transA` / `transB` are `'N'`, `'T'`, or `'C'`.
"""
function sp2m_nnz_count(transA, A::MKLSparseMatrix{S}, descrA::matrix_descr,
                        transB, B::MKLSparseMatrix{S}, descrB::matrix_descr
) where S <: AbstractSparseMatrix
    opA = convert(sparse_operation_t, transA)
    opB = convert(sparse_operation_t, transB)
    Cout = Ref{sparse_matrix_t}()
    res = mkl_call(Val{:mkl_sparse_sp2mI}(), S,
                   opA, descrA, A.handle, opB, descrB, B.handle,
                   SPARSE_STAGE_NNZ_COUNT, Cout)
    check_status(res)
    return MKLSparseMatrix{S}(Cout[])
end

"""
    sp2m_finalize_mult!(C::MKLSparseMatrix{S},
                        transA, A::MKLSparseMatrix{S}, descrA,
                        transB, B::MKLSparseMatrix{S}, descrB) -> C

Stage 2 of the two-stage `sp2m`: fill in the numeric values of `C = op(A) * op(B)`
into the existing pattern computed by `sp2m_nnz_count`.

`C` is mutated in place. `transA` / `transB` and `descrA` / `descrB` must match
the values used in the corresponding `sp2m_nnz_count` call.
"""
function sp2m_finalize_mult!(C::MKLSparseMatrix{S},
                             transA, A::MKLSparseMatrix{S}, descrA::matrix_descr,
                             transB, B::MKLSparseMatrix{S}, descrB::matrix_descr
) where S <: AbstractSparseMatrix
    opA = convert(sparse_operation_t, transA)
    opB = convert(sparse_operation_t, transB)
    Cref = Ref(C.handle)
    res = mkl_call(Val{:mkl_sparse_sp2mI}(), S,
                   opA, descrA, A.handle, opB, descrB, B.handle,
                   SPARSE_STAGE_FINALIZE_MULT, Cref)
    check_status(res)
    return C
end

"""
    sp2m_full_mult(transA, A::MKLSparseMatrix{S}, descrA,
                   transB, B::MKLSparseMatrix{S}, descrB) -> hC::MKLSparseMatrix{S}

One-shot `sp2m` (`SPARSE_STAGE_FULL_MULT`): compute pattern and values together.
The caller owns the returned handle. Provided for completeness so users can
benchmark the monolithic path against the split path on the same inputs.
"""
function sp2m_full_mult(transA, A::MKLSparseMatrix{S}, descrA::matrix_descr,
                        transB, B::MKLSparseMatrix{S}, descrB::matrix_descr
) where S <: AbstractSparseMatrix
    opA = convert(sparse_operation_t, transA)
    opB = convert(sparse_operation_t, transB)
    Cout = Ref{sparse_matrix_t}()
    res = mkl_call(Val{:mkl_sparse_sp2mI}(), S,
                   opA, descrA, A.handle, opB, descrB, B.handle,
                   SPARSE_STAGE_FULL_MULT, Cout)
    check_status(res)
    return MKLSparseMatrix{S}(Cout[])
end
