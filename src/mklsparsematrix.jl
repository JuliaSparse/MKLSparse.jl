## MKL sparse matrix

# https://github.com/JuliaSmoothOptimizers/SparseMatricesCOO.jl
mutable struct SparseMatrixCOO{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    rows::Vector{Ti}
    cols::Vector{Ti}
    vals::Vector{Tv}
end

# https://github.com/gridap/SparseMatricesCSR.jl
mutable struct SparseMatrixCSR{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    rowptr::Vector{Ti}
    colval::Vector{Ti}
    nzval::Vector{Tv}
end

mkl_storagetype_specifier(::Type{<:SparseMatrixCOO}) = "coo"
mkl_storagetype_specifier(::Type{<:SparseMatrixCSR}) = "csr"

Base.size(A::MKLSparse.SparseMatrixCOO) = (A.m, A.n)
Base.size(A::MKLSparse.SparseMatrixCSR) = (A.m, A.n)

SparseArrays.nnz(A::MKLSparse.SparseMatrixCOO) = length(A.vals)
SparseArrays.nnz(A::MKLSparse.SparseMatrixCSR) = length(A.nzval)

matrix_descr(A::MKLSparse.SparseMatrixCSR) = matrix_descr('G', 'F', 'N')
matrix_descr(A::MKLSparse.SparseMatrixCOO) = matrix_descr('G', 'F', 'N')

mutable struct MKLSparseMatrix
    handle::sparse_matrix_t
end

Base.unsafe_convert(::Type{sparse_matrix_t}, desc::MKLSparseMatrix) = desc.handle

function MKLSparseMatrix(A::SparseMatrixCOO, IndexBase::Char='O')
    descr_ref = Ref{sparse_matrix_t}()
    mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
             descr_ref, IndexBase, A.m, A.n, nnz(A), A.rows, A.cols, A.vals,
             log=Val{false}())
    obj = MKLSparseMatrix(descr_ref[])
    finalizer(mkl_function(Val{:mkl_sparse_destroyI}(), typeof(A)), obj)
    return obj
end

function MKLSparseMatrix(A::SparseMatrixCSR, IndexBase::Char='O')
    descr_ref = Ref{sparse_matrix_t}()
    mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
             descr_ref, IndexBase, A.m, A.n, A.rowptr, pointer(A.rowptr, 2), A.colval, A.nzval,
             log=Val{false}())
    obj = MKLSparseMatrix(descr_ref[])
    finalizer(mkl_function(Val{:mkl_sparse_destroyI}(), typeof(A)), obj)
    return obj
end

function MKLSparseMatrix(A::SparseMatrixCSC, IndexBase::Char='O')
    descr_ref = Ref{sparse_matrix_t}()
    mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
             descr_ref, IndexBase, A.m, A.n, A.colptr, pointer(A.colptr, 2), A.rowval, A.nzval,
             log=Val{false}())
    obj = MKLSparseMatrix(descr_ref[])
    finalizer(mkl_function(Val{:mkl_sparse_destroyI}(), typeof(A)), obj)
    return obj
end
