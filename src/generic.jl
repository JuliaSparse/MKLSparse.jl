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
    res = mkl_call(Val{:mkl_sparse_T_mvI}(), typeof(A),
             transA, alpha, MKLSparseMatrix(A), descr, x, beta, y)
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
    res = mkl_call(Val{:mkl_sparse_T_mmI}(), typeof(A),
                   transA, alpha, MKLSparseMatrix(A), descr, dense_layout, B, columns, ldB, beta, C, ldC)
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
    res = mkl_call(Val{:mkl_sparse_T_trsvI}(), typeof(A),
                   transA, alpha, MKLSparseMatrix(A), descr, x, y)
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
    res = mkl_call(Val{:mkl_sparse_T_trsmI}(), typeof(A),
                   transA, alpha, MKLSparseMatrix(A), descr, dense_layout, X, columns, ldX, Y, ldY)
    check_status(res)
    return Y
end
