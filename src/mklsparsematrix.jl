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

function Base.convert(::Type{Array}, spA::MKLSparse.SparseMatrixCOO{T}) where T
    A = fill(zero(T), spA.m, spA.n)
    for (i, j, v) in zip(spA.rows, spA.cols, spA.vals)
        A[i, j] = v
    end
    return A
end

Base.convert(::Type{Array}, spA::MKLSparse.SparseMatrixCSR) =
    convert(Array, transpose(convert(SparseMatrixCSC, transpose(spA))))

# lazypermutedims(sparse) does not do in any new array allocations,
# it just switches the layout of the sparse matrix reusing the same data

# transpose the sparse matrix and switch its layout from CSC to CSR
lazypermutedims(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} =
    convert(SparseMatrixCSR{Tv, Ti}, transpose(A))

# transpose the sparse matrix and switch its layout from CSR to CSC
lazypermutedims(A::SparseMatrixCSR{Tv, Ti}) where {Tv, Ti} =
    convert(SparseMatrixCSC{Tv, Ti}, transpose(A))

# transpose the sparse matrix in COO
lazypermutedims(A::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti} =
    SparseMatrixCOO{Tv, Ti}(A.n, A.m, A.cols, A.rows, A.vals)

# when lazypermutedims() is applied to matrix A, its description should be updated
lazypermutedims(descr::matrix_descr) = matrix_descr(
    descr.type,
    descr.mode == SPARSE_FILL_MODE_UPPER ? SPARSE_FILL_MODE_LOWER :
    descr.mode == SPARSE_FILL_MODE_LOWER ? SPARSE_FILL_MODE_UPPER : descr.mode,
    descr.diag)

"""
    MKLSparseMatrix{S}

A wrapper around the handle of a MKLSparse matrix
created from the Julia sparse matrix of type `S`.
"""
struct MKLSparseMatrix{S <: AbstractSparseMatrix}
    handle::sparse_matrix_t
end

Base.unsafe_convert(::Type{sparse_matrix_t}, A::MKLSparseMatrix) = A.handle

# create sparse_matrix_t handle for the SparseMKL representation of a given sparse matrix
# the created SparseMKL matrix handle has to be disposed by calling destroy_handle()
function MKLSparseMatrix(A::SparseMatrixCOO; index_base = SPARSE_INDEX_BASE_ONE)
    ref = Ref{sparse_matrix_t}()
    res = mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
                   ref, index_base, A.m, A.n, nnz(A), A.rows, A.cols, A.vals,
                   log=Val{false}())
    check_status(res)
    return MKLSparseMatrix{typeof(A)}(ref[])
end

function MKLSparseMatrix(A::SparseMatrixCSR; index_base = SPARSE_INDEX_BASE_ONE)
    ref = Ref{sparse_matrix_t}()
    res = mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
                   ref, index_base, A.m, A.n, A.rowptr, pointer(A.rowptr, 2), A.colval, A.nzval,
                   log=Val{false}())
    check_status(res)
    return MKLSparseMatrix{typeof(A)}(ref[])
end

function MKLSparseMatrix(A::SparseMatrixCSC; index_base = SPARSE_INDEX_BASE_ONE)
    ref = Ref{sparse_matrix_t}()
    # SparseMatrixCSC is fixed to 1-based indexing, passing SPARSE_INDEX_BASE_ZERO is most likely an error
    res = mkl_call(Val{:mkl_sparse_T_create_SI}(), typeof(A),
                   ref, index_base, A.m, A.n, A.colptr, pointer(A.colptr, 2), A.rowval, A.nzval,
                   log=Val{false}())
    check_status(res)
    return MKLSparseMatrix{typeof(A)}(ref[])
end

function destroy(A::MKLSparseMatrix{S}) where S
    if A.handle != C_NULL
        res = mkl_call(Val{:mkl_sparse_destroyI}(), S, A.handle, log=Val{false}())
        check_status(res)
        return res
    else
        return SPARSE_STATUS_NOT_INITIALIZED
    end
end

# extract the Intel MKL's sparse matrix A information assuming its storage type is S
# the returned arrays are internal to MKL representation of A, their lifetime is limited by A
# "major_" refers to the major axis (rows for CSR, columns for CSC)
# "minor_" refers to the minor axis (columns for CSR, rows for CSC)
function extract_data(ref::MKLSparseMatrix{S}) where {S <: AbstractSparseMatrix{Tv, Ti}} where {Tv, Ti}
    IT = ifelse(BlasInt === Int64 && Ti === Int32, BlasInt, Ti)
    index_base = Ref{sparse_index_base_t}()
    nrows = Ref{IT}(0)
    ncols = Ref{IT}(0)
    major_startsptr = Ref{Ptr{IT}}()
    major_endsptr = Ref{Ptr{IT}}()
    minor_valptr = Ref{Ptr{IT}}()
    nzvalptr = Ref{Ptr{Tv}}()
    res = mkl_call(Val{:mkl_sparse_T_export_SI}(), S,
                   ref, index_base, nrows, ncols, major_startsptr, major_endsptr, minor_valptr, nzvalptr,
                   log=Val{false}())
    check_status(res)
    nmajor = S <: SparseMatrixCSC ? ncols[] : S <: SparseMatrixCSR ? nrows[] : error("Unsupported storage type $S")
    if major_startsptr[] != C_NULL && major_endsptr[] != C_NULL
        major_starts = unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, major_startsptr[]), nmajor, own=false)
        major_ends = unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, major_endsptr[]), nmajor, own=false)
        @assert major_ends[end] >= major_starts[1]
        # check if minor_val and nzvl values occupy continuous memory segment
        if pointer(major_ends) == pointer(major_starts, 2) # all(major_starts[i + 1] == major_ends[i] for i in 1:length(major_starts)-1)
            major_starts = unsafe_wrap(Vector{Ti}, pointer(major_starts), nmajor + 1, own=false)
        else
            error("Support for non-continuous minor axis indices and non-zero values is not implemented")
        end
    else
        major_starts = nothing
        major_ends = nothing
    end
    return (
        size = (nrows[], ncols[]),
        index_base = index_base[],
        major_starts = major_starts,
        #major_ends = major_ends,
        minor_val = minor_valptr[] != C_NULL ? unsafe_wrap(Vector{Ti}, reinterpret(Ptr{Ti}, minor_valptr[]),
                                                           major_ends[end] - major_starts[1], own=false) : nothing,
        nzval = nzvalptr[] != C_NULL ? unsafe_wrap(Vector{Tv}, nzvalptr[],
                                                   major_ends[end] - major_starts[1], own=false) : nothing,
    )
end

# check that source and destination have the same non-zero structure
function check_nzpattern(dest::AbstractSparseMatrix, src::NamedTuple)
    src_nnz = !isnothing(src.major_starts) ? src.major_starts[end] - 1 : 0
    nnz(dest) == src_nnz ||
        error(lazy"Number of nonzeros in the destination matrix ($(nnz(dest))) does not match the source ($(src_nnz))")

    dest_major_starts = dest isa SparseMatrixCSC ? dest.colptr :
                        dest isa SparseMatrixCSR ? dest.rowptr :
                        error(lazy"Unsupported storage type $(typeof(dest))")
    isnothing(src.major_starts) || dest_major_starts == src.major_starts ||
        error("Nonzeros structure of the destination matrix does not match the source (major starts)")

    Ti = eltype(src.minor_val)
    dest_minor_val = dest isa SparseMatrixCSC ? dest.rowval :
                     dest isa SparseMatrixCSR ? dest.colval :
                     error(lazy"Unsupported storage type $(typeof(dest))")
    # skip minor_val check if not provided
    if !isnothing(src.minor_val)
        # convert minor_vals to 1-based if the source is 0-based and the destination is SparseMatrixCSC
        minors_match = src.index_base == SPARSE_INDEX_BASE_ZERO && dest isa SparseMatrixCSC ?
            all((a, b) -> a + one(Ti) == b, zip(src.minor_val, dest_minor_val)) : # convert to 1-based
            src.minor_val == dest_minor_val
        minors_match ||
            error("Nonzeros structure of the destination matrix does not match the source (minor values)")
    end
end

check_nzpattern(dest::S, src::MKLSparseMatrix{S}) where S <: AbstractSparseMatrix =
    check_nzpattern(dest, extract_data(src))

function Base.convert(::Type{S}, A::MKLSparseMatrix{S}) where {S <: SparseMatrixCSC{Tv, Ti}} where {Tv, Ti}
    _A = extract_data(A)
    colptr = !isnothing(_A.major_starts) ?
        Vector{Ti}(_A.major_starts) :
        fill(one(Ti), _A.size[2] + 1)
    rowval = !isnothing(_A.minor_val) ?
        _A.index_base == SPARSE_INDEX_BASE_ZERO ?
        [Ti(v) + one(Ti) for v in _A.minor_val] : # convert to 1-based
        Vector{Ti}(_A.minor_val) :
        Ti[]
    nzval = !isnothing(_A.nzval) ? Vector{Tv}(_A.nzval) : Tv[]
    return S(_A.size..., colptr, rowval, nzval)
end

# converter for the default SparseMatrixCSC storage type
Base.convert(::Type{SparseMatrixCSC}, A::MKLSparseMatrix{SparseMatrixCSC{Tv, Ti}}) where {Tv, Ti} =
    convert(SparseMatrixCSC{Tv, Ti}, A)

function Base.convert(::Type{S}, A::MKLSparseMatrix{S}) where {S <: SparseMatrixCSR{Tv, Ti}} where {Tv, Ti}
    _A = extract_data(A)
    rowptr = !isnothing(_A.major_starts) ?
        Vector{Ti}(_A.major_starts) :
        fill(one(Ti), _A.size[1] + 1)
    # not converting the col indices depending on index_base
    colval = !isnothing(_A.minor_val) ? Vector{Ti}(_A.minor_val) : Ti[]
    nzval = !isnothing(_A.nzval) ? Vector{Tv}(_A.nzval) : Tv[]
    return S(_A.size..., rowptr, colval, nzval)
end

Base.convert(::Type{SparseMatrixCSR}, A::MKLSparseMatrix{SparseMatrixCSR{Tv, Ti}}) where {Tv, Ti} =
    convert(SparseMatrixCSR{Tv, Ti}, A)
