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

function mv!(transa::Char, alpha::T, A::AbstractSparseMatrix{T}, descr::matrix_descr, x::StridedVector{T}, beta::T, y::StridedVector{T}) where T
    check_transa(transa)
    check_mat_op_sizes(y, A, transa, x, 'N')
    mkl_call(Val{:mkl_sparse_T_mvI}(), typeof(A),
             transa, alpha, MKLSparseMatrix(A), descr, x, beta, y)
    return y
end

function mm!(transa::Char, alpha::T, A::AbstractSparseMatrix{T}, descr::matrix_descr, x::StridedMatrix{T}, beta::T, y::StridedMatrix{T}) where T
    check_transa(transa)
    check_mat_op_sizes(y, A, transa, x, 'N')
    columns = size(y, 2)
    ldx = stride(x, 2)
    ldy = stride(y, 2)
    mkl_call(Val{:mkl_sparse_T_mmI}(), typeof(A),
             transa, alpha, MKLSparseMatrix(A), descr, 'C', x, columns, ldx, beta, y, ldy)
    return y
end

function trsv!(transa::Char, alpha::T, A::AbstractSparseMatrix{T}, descr::matrix_descr, x::StridedVector{T}, y::StridedVector{T}) where T
    checksquare(A)
    check_transa(transa)
    check_mat_op_sizes(y, A, transa, x, 'N')
    mkl_call(Val{:mkl_sparse_T_trsvI}(), typeof(A),
             transa, alpha, MKLSparseMatrix(A), descr, x, y)
    return y
end

function trsm!(transa::Char, alpha::T, A::AbstractSparseMatrix{T}, descr::matrix_descr, x::StridedMatrix{T}, y::StridedMatrix{T}) where T
    checksquare(A)
    check_transa(transa)
    check_mat_op_sizes(y, A, transa, x, 'N')
    columns = size(y, 2)
    ldx = stride(x, 2)
    ldy = stride(y, 2)
    mkl_call(Val{:mkl_sparse_T_trsmI}(), typeof(A),
             transa, alpha, MKLSparseMatrix(A), descr, 'C', x, columns, ldx, y, ldy)
    return y
end
