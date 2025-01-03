# Intermediate wrappers for the Sparse BLAS routines
# that check the parameters validity (including matrix dimensions checks)
# and convert Julia's matrix types to the MKL's matrix types.
# See https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/inspector-executor-sparse-blas-execution-routines.html
# for the detailed description of the wrapped functions.

# generates the reference to the MKL function from the template
@inline @generated function mkl_function(
    ::Val{F}, ::Type{S}
) where S <: AbstractSparseMatrix{Tv, Ti} where {F, Tv, Ti}
    mkl_function_name(F, S, Tv, Ti)
end

# calls MKL function with the name constructed from the template F (e.g. :mkl_Tcscmm)
# using the sparse matrix type S (e.g. SparseMatrixCSC{Float64, Int32}),
# see mkl_function_name()
@inline @generated function mkl_call(
    ::Val{F}, ::Type{S}, args...;
    log::Val{L} = Val{true}()
) where {L, S <: AbstractSparseMatrix{Tv, Ti}} where {F, Tv, Ti}
    fname = mkl_function_name(F, S, Tv, Ti)
    body = Expr(:call, fname, (:(args[$i]) for i in eachindex(args))...)
    L && (body = Expr(:block, :(_log_mklsparse_call($(QuoteNode(fname)))), body))
    return body
end

# y := alpha * op(A) * x + beta * y
function mv!(transA::Char, alpha::T, A::AbstractSparseMatrix{T}, descr::matrix_descr,
             x::StridedVector{T}, beta::T, y::StridedVector{T}
) where T
    check_trans(transA)
    check_mat_op_sizes(y, A, transA, x, 'N')
    hA = MKLSparseMatrix(A)
    res = mkl_call(Val{:mkl_sparse_T_mvI}(), typeof(A),
             transA, alpha, hA, descr, x, beta, y)
    destroy(hA)
    check_status(res)
    return y
end

# C := alpha * op(A) * B + beta * C
function mm!(transA::Char, alpha::T, A::AbstractSparseMatrix{T}, descr::matrix_descr,
             B::StridedMatrix{T}, beta::T, C::StridedMatrix{T};
             dense_layout::sparse_layout_t = SPARSE_LAYOUT_COLUMN_MAJOR
) where T
    check_trans(transA)
    check_mat_op_sizes(C, A, transA, B, 'N'; dense_layout)
    columns = size(C, dense_layout == SPARSE_LAYOUT_COLUMN_MAJOR ? 2 : 1)
    ldB = stride(B, 2)
    ldC = stride(C, 2)
    hA = MKLSparseMatrix(A)
    res = mkl_call(Val{:mkl_sparse_T_mmI}(), typeof(A),
                   transA, alpha, hA, descr, dense_layout, B, columns, ldB, beta, C, ldC)
    destroy(hA)
    check_status(res)
    return C
end

# C := opA(A) * opB(B), where C is sparse
function sp2m(transA::Char, A::S, descrA::matrix_descr,
              transB::Char, B::S, descrB::matrix_descr
) where S <: AbstractSparseMatrix
    check_trans(transA)
    check_trans(transB)
    check_mat_op_sizes(nothing, A, transA, B, transB)
    Cout = Ref{sparse_matrix_t}()
    hA = MKLSparseMatrix(A)
    hB = MKLSparseMatrix(B)
    res = mkl_call(Val{:mkl_sparse_sp2mI}(), S,
                   transA, descrA, hA, transB, descrB, hB,
                   SPARSE_STAGE_FULL_MULT, Cout)
    destroy(hA)
    destroy(hB)
    check_status(res)
    # NOTE: we are guessing what is the storage format of C
    hC = MKLSparseMatrix{S}(Cout[])
    C = convert(S, hC)
    destroy(hC)
    return C
end

# C := opA(A) * opB(B), where C is sparse, in-place version
# C should have the correct size and sparsity pattern
function sp2m!(transA::Char, A::S, descrA::matrix_descr,
               transB::Char, B::S, descrB::matrix_descr,
               C::S;
               check_nzpattern::Bool = true
) where {S <: AbstractSparseMatrix}
    check_trans(transA)
    check_trans(transB)
    check_mat_op_sizes(C, A, transA, B, transB)
    hA = MKLSparseMatrix(A)
    hB = MKLSparseMatrix(B)
    if check_nzpattern
        # pre-multiply A * B to get the number of nonzeros per column in the result
        CptnOut = Ref{sparse_matrix_t}()
        res = mkl_call(Val{:mkl_sparse_sp2mI}(), S,
                    transA, descrA, hA, transB, descrB, hB,
                    SPARSE_STAGE_NNZ_COUNT, CptnOut)
        check_status(res)
        hCptn = MKLSparseMatrix{S}(CptnOut[])
        try
            # check if C has the same per-column nonzeros as the result
            MKLSparse.check_nzpattern(C, hCptn)
        catch e
            # destroy handles to A and B if the pattern check fails,
            # otherwise reuse them at the actual multiplication
            destroy(hA)
            destroy(hB)
            rethrow(e)
        finally
            destroy(hCptn)
        end
        # FIXME rowval not checked
    end
    # FIXME the optimal way would be to create the MKLSparse handle to C reusing its arrays
    # and do SPARSE_STAGE_FINALIZE_MULT to directly write to the C.nzval
    # but that causes segfaults when the handle is destroyed
    # (also the partial mkl_sparse_copy(C) workaround to reuse the nz structure segfaults)
    # see the note stating that external memory is not currently supported:
    # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/two-stage-algorithm-in-inspect-execute-sp-blas.html
    #hC = MKLSparseMatrix(C)
    #mkl_call(Val{:mkl_sparse_set_memory_hintI}(), typeof(C), SPARSE_MEMORY_NONE)
    #hC_ref = Ref(hC)
    #res = mkl_call(Val{:mkl_sparse_sp2mI}(), typeof(A),
    #               transA, descrA, hA, transB, descrB, hB,
    #               SPARSE_STAGE_FINALIZE_MULT, hC_ref)
    #@assert hC_ref[] == hC
    # so instead we do the full multiplication and copy the result into C nzvals
    hCopy_ref = Ref{sparse_matrix_t}()
    # don't log if it's the 2nd mkl_call
    res = mkl_call(Val{:mkl_sparse_sp2mI}(), S,
                   transA, descrA, hA, transB, descrB, hB,
                   SPARSE_STAGE_FULL_MULT, hCopy_ref, log = Val(!check_nzpattern))
    destroy(hA)
    destroy(hB)
    #destroy(hC)
    check_status(res)
    if hCopy_ref[] != C_NULL
        hCopy = MKLSparseMatrix{S}(hCopy_ref[])
        copy!(C, hCopy; check_nzpattern)
        destroy(hCopy)
    end
    return C
end

# C := alpha * opA(A) * opB(B) + beta * C, where C is dense
function sp2md!(transA::Char, alpha::T, A::S, descrA::matrix_descr,
                transB::Char, B::S, descrB::matrix_descr,
                beta::T, C::StridedMatrix{T};
                dense_layout::sparse_layout_t = SPARSE_LAYOUT_COLUMN_MAJOR
) where {S <: AbstractSparseMatrix{T}} where T
    check_trans(transA)
    check_trans(transB)
    check_mat_op_sizes(C, A, transA, B, transB)
    ldC = stride(C, 2)
    hA = MKLSparseMatrix(A)
    hB = MKLSparseMatrix(B)
    res = mkl_call(Val{:mkl_sparse_T_sp2mdI}(), S,
                   transA, descrA, hA, transB, descrB, hB,
                   alpha, beta,
                   C, dense_layout, ldC)
    destroy(hA)
    destroy(hB)
    check_status(res)
    return C
end

# find y: op(A) * y = alpha * x
function trsv!(transA::Char, alpha::T, A::AbstractSparseMatrix{T}, descr::matrix_descr,
               x::StridedVector{T}, y::StridedVector{T}
) where T
    checksquare(A)
    check_trans(transA)
    check_mat_op_sizes(y, A, transA, x, 'N')
    hA = MKLSparseMatrix(A)
    res = mkl_call(Val{:mkl_sparse_T_trsvI}(), typeof(A),
                   transA, alpha, hA, descr, x, y)
    destroy(hA)
    check_status(res)
    return y
end

# Y := alpha * inv(op(A)) * X
function trsm!(transA::Char, alpha::T, A::AbstractSparseMatrix{T}, descr::matrix_descr,
               X::StridedMatrix{T}, Y::StridedMatrix{T};
               dense_layout::sparse_layout_t = SPARSE_LAYOUT_COLUMN_MAJOR
) where T
    checksquare(A)
    check_trans(transA)
    check_mat_op_sizes(Y, A, transA, X, 'N'; dense_layout)
    columns = size(Y, dense_layout == SPARSE_LAYOUT_COLUMN_MAJOR ? 2 : 1)
    ldX = stride(X, 2)
    ldY = stride(Y, 2)
    hA = MKLSparseMatrix(A)
    res = mkl_call(Val{:mkl_sparse_T_trsmI}(), typeof(A),
                   transA, alpha, hA, descr, dense_layout, X, columns, ldX, Y, ldY)
    destroy(hA)
    check_status(res)
    return Y
end
