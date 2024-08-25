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

Base.:(==)(A::MKLSparse.SparseMatrixCOO, B::MKLSparse.SparseMatrixCOO) =
    A.m == B.m && A.n == B.n && A.rows == B.rows && A.cols == B.cols && A.vals == B.vals

Base.:(==)(A::MKLSparse.SparseMatrixCSR, B::MKLSparse.SparseMatrixCSR) =
    A.m == B.m && A.n == B.n && A.rowptr == B.rowptr && A.colval == B.colval && A.nzval == B.nzval

Base.convert(::Type{SparseMatrixCSR{Tv, Ti}}, tA::Transpose{Tv, SparseMatrixCSC{Tv, Ti}}) where {Tv, Ti} =
    SparseMatrixCSR{Tv, Ti}(size(tA)..., parent(tA).colptr, rowvals(parent(tA)), nonzeros(parent(tA)))

Base.convert(::Type{SparseMatrixCSR}, tA::Transpose{Tv, SparseMatrixCSC{Tv, Ti}}) where {Tv, Ti} =
    convert(SparseMatrixCSR{Tv, Ti}, tA)

Base.convert(::Type{SparseMatrixCSC{Tv, Ti}}, tA::Transpose{Tv, SparseMatrixCSR{Tv, Ti}}) where {Tv, Ti} =
    SparseMatrixCSC{Tv, Ti}(size(tA)..., parent(tA).rowptr, parent(tA).colval, parent(tA).nzval)

Base.convert(::Type{SparseMatrixCSC}, tA::Transpose{Tv, SparseMatrixCSR{Tv, Ti}}) where {Tv, Ti} =
    convert(SparseMatrixCSC{Tv, Ti}, tA)

function Base.convert(::Type{SparseMatrixCOO{Tv, Ti}}, A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    rows, cols, vals = findnz(A)
    return SparseMatrixCOO{Tv, Ti}(size(A)..., convert(Vector{Ti}, rows), convert(Vector{Ti}, cols), vals)
end

Base.convert(::Type{SparseMatrixCOO}, A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} =
    convert(SparseMatrixCOO{Tv, Ti}, A)

"""
    MKLSparseMatrix

A wrapper around a MKLSparse matrix handle.
"""
mutable struct MKLSparseMatrix
    handle::sparse_matrix_t
end

Base.unsafe_convert(::Type{sparse_matrix_t}, desc::MKLSparseMatrix) = desc.handle

function MKLSparseMatrix(A::SparseMatrixCOO; index_base = SPARSE_INDEX_BASE_ONE)
    matrix_ref = Ref{sparse_matrix_t}()
    res = mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
                   matrix_ref, index_base, A.m, A.n, nnz(A), A.rows, A.cols, A.vals,
                   log=Val{false}())
    check_status(res)
    obj = MKLSparseMatrix(matrix_ref[])
    finalizer(mkl_function(Val{:mkl_sparse_destroyI}(), typeof(A)), obj)
    return obj
end

function MKLSparseMatrix(A::SparseMatrixCSR; index_base = SPARSE_INDEX_BASE_ONE)
    matrix_ref = Ref{sparse_matrix_t}()
    res = mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
                   matrix_ref, index_base, A.m, A.n, A.rowptr, pointer(A.rowptr, 2), A.colval, A.nzval,
                   log=Val{false}())
    check_status(res)
    obj = MKLSparseMatrix(matrix_ref[])
    finalizer(mkl_function(Val{:mkl_sparse_destroyI}(), typeof(A)), obj)
    return obj
end

function MKLSparseMatrix(A::SparseMatrixCSC; index_base = SPARSE_INDEX_BASE_ONE)
    # SparseMatrixCSC is fixed to 1-based indexing, passing SPARSE_INDEX_BASE_ZERO is most likely an error
    matrix_ref = Ref{sparse_matrix_t}()
    res = mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
                   matrix_ref, index_base, A.m, A.n, A.colptr, pointer(A.colptr, 2), A.rowval, A.nzval,
                   log=Val{false}())
    check_status(res)
    obj = MKLSparseMatrix(matrix_ref[])
    finalizer(mkl_function(Val{:mkl_sparse_destroyI}(), typeof(A)), obj)
    return obj
end

function Base.convert(::Type{S}, A::MKLSparseMatrix) where {S <: SparseMatrixCSC{Tv, Ti}} where {Tv, Ti}
    IT = ifelse(BlasInt === Int64 && Ti === Int32, BlasInt, Ti)
    index_base = Ref{sparse_index_base_t}()
    nrows = Ref{IT}(0)
    ncols = Ref{IT}(0)
    colstartsptr = Ref{Ptr{IT}}()
    colendsptr = Ref{Ptr{IT}}()
    rowvalptr = Ref{Ptr{IT}}()
    nzvalptr = Ref{Ptr{Tv}}()
    res = mkl_call(Val{:mkl_sparse_T_export_SI}(), S,
                   A.handle, index_base, nrows, ncols, colstartsptr, colendsptr, rowvalptr, nzvalptr,
                   log=Val{false}())
    check_status(res)
    colstarts = unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, colstartsptr[]), ncols[], own=false)
    colends = unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, colendsptr[]), ncols[], own=false)
    @assert colends[end] >= colstarts[1]
    rowval = unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, rowvalptr[]), colends[end] - colstarts[1], own=false)
    nzval = unsafe_wrap(Vector{Tv}, nzvalptr[], colends[end] - colstarts[1], own=false)
    rowval = if index_base[] == SPARSE_INDEX_BASE_ZERO
        rowval .+ one(Ti) # convert to 1-based (rowval is copied)
    else
        copy(rowval)
    end
    # check if row and nz values occupy continuous memory segment
    if pointer(colends) == pointer(colstarts, 2) # all(colstarts[i + 1] == colends[i] for i in 1:length(colstarts)-1)
        colstarts = unsafe_wrap(Vector{Ti}, pointer(colstarts), ncols[] + 1, own=false)
        return S(nrows[], ncols[], copy(colstarts), rowval, copy(nzval))
    else
        error("Support for non-continuous row and values is not implemented")
    end
end

# converter for the default SparseMatrixCSC storage type
Base.convert(::Type{SparseMatrixCSC}, A::MKLSparseMatrix) =
    convert(SparseMatrixCSC{Float64, BlasInt}, A)

function Base.convert(::Type{S}, A::MKLSparseMatrix) where {S <: SparseMatrixCSR{Tv, Ti}} where {Tv, Ti}
    IT = ifelse(BlasInt === Int64 && Ti === Int32, BlasInt, Ti)
    index_base = Ref{sparse_index_base_t}()
    nrows = Ref{IT}(0)
    ncols = Ref{IT}(0)
    rowstartsptr = Ref{Ptr{IT}}()
    rowendsptr = Ref{Ptr{IT}}()
    colvalptr = Ref{Ptr{IT}}()
    nzvalptr = Ref{Ptr{Tv}}()
    res = mkl_call(Val{:mkl_sparse_T_export_SI}(), S,
                   A.handle, index_base, nrows, ncols, rowstartsptr, rowendsptr, colvalptr, nzvalptr,
                   log=Val{false}())
    check_status(res)
    rowstarts = unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, rowstartsptr[]), nrows[], own=false)
    rowends = unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, rowendsptr[]), nrows[], own=false)
    @assert rowends[end] >= rowstarts[1]
    colval = unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, colvalptr[]), rowends[end] - rowstarts[1], own=false)
    nzval = unsafe_wrap(Vector{Tv}, nzvalptr[], rowends[end] - rowstarts[1], own=false)
    # not converting the col indices depending on index_base
    # check if row and nz values occupy continuous memory segment
    if pointer(rowends) == pointer(rowstarts, 2) # all(rowstarts[i + 1] == rowends[i] for i in 1:length(rowstarts)-1)
        rowstarts = unsafe_wrap(Vector{Ti}, pointer(rowstarts), nrows[] + 1, own=false)
        return S(nrows[], ncols[], copy(rowstarts), copy(colval), copy(nzval))
    else
        error("Support for non-continuous row and values is not implemented")
    end
end
