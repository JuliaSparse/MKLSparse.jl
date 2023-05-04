function _check_transa(t::Char)
    if !(t in ('C', 'N', 'T'))
        throw(ArgumentError("transa: is '$t', must be 'N', 'T', or 'C'"))
    end
end

# Checks sizes for the multiplication C <- tA[A] * tB[B]
function _check_mat_mult_matvec(C, A, tA, B, tB)
    _size(t::Char, M::AbstractMatrix) = t == 'N' ? size(M) : reverse(size(M))
    _size(t::Char, V::AbstractVector) = t == 'N' ? (size(V, 1), 1) : (1, size(V, 1))
    _str(M::AbstractMatrix) = string("[", size(M, 1), ", ", size(M, 2), "]")
    _str(V::AbstractVector) = string("[", size(V, 1), "]")
    _t(t) = t == 'T' ? "ᵀ" : t == 'C' ? "ᴴ" : t == 'N' ? "" : "ERROR"

    mA, nA = _size(tA, A)
    mB, nB = _size(tB, B)
    mC, nC = _size('N', C)
    if nA != mB || mC != mA || nC != nB
        str = string("arrays had inconsistent dimensions for C = A", _t(tA), " * B", _t(tB), ": ",
                     _str(C), " = ", _str(A), _t(tA), " * ", _str(B), _t(tB))
        throw(DimensionMismatch(str))
    end
end

# MKL convention by annotating the type of numeric arguments in method names
mkl_typespec(::Type{T}) where T =
    T == Float32 ? "s" :
    T == Float64 ? "d" :
    T == ComplexF32 ? "c" :
    T == ComplexF64 ? "z" :
    throw(ArgumentError("Unsupported type $(T)"))

# calls MKL function with the name template F (e.g. :mkl_Tcscmm),
# with 'T' char replaced by a type-specifier corresponding to the input type T,
# (e.g. 's' for Float32), so the function called is :mkl_scscmm
@inline @generated function mkl_call(::Val{F}, ::Type{T}, args...) where {F, T}
    fname = Symbol(replace(String(F), "T" => mkl_typespec(T)))
    quote
        _log_mklsparse_call($fname)
        $fname(args...)
    end
end

# same but doesn't log the call
@inline @generated function mkl_call_nolog(::Val{F}, ::Type{T}, args...) where {F, T}
    fname = Symbol(replace(String(F), "T" => mkl_typespec(T)))
    :($fname(args...))
end

function cscmv!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, BlasInt}, x::StridedVector{T},
                β::T, y::StridedVector{T}) where {T <: BlasFloat}
    _check_transa(transa)
    _check_mat_mult_matvec(y, A, transa, x, 'N')

    mkl_call(Val(:mkl_Tcscmv), T, transa, A.m, A.n, α, matdescra,
             A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y)
    return y
end

function cscmm!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, BlasInt}, B::StridedMatrix{T},
                β::T, C::StridedMatrix{T}) where {T <: BlasFloat}
    _check_transa(transa)
    _check_mat_mult_matvec(C, A, transa, B, 'N')
    mB, nB = size(B)
    mC, nC = size(C)

    mkl_call(Val(:mkl_Tcscmm), T, transa, A.m, nC, A.n, α, matdescra,
             A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, β, C, mC)
    return C
end

function cscsv!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, BlasInt}, x::StridedVector{T},
                y::StridedVector{T}) where {T <: BlasFloat}
    n = checksquare(A)
    _check_transa(transa)
    _check_mat_mult_matvec(y, A, transa, x, 'N')

    mkl_call(Val(:mkl_Tcscsv), T, transa, A.m, α, matdescra,
             A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y)
    return y
end

function cscsm!(transa::Char, α::T, matdescra::String,
                A::SparseMatrixCSC{T, BlasInt}, B::StridedMatrix{T},
                C::StridedMatrix{T}) where {T <: BlasFloat}
    mB, nB = size(B)
    mC, nC = size(C)
    n = checksquare(A)
    _check_transa(transa)
    _check_mat_mult_matvec(C, A, transa, B, 'N')

    mkl_call(Val(:mkl_Tcscsm), T, transa, A.n, nC, α, matdescra,
             A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC)
    return C
end

# creates MKL sparse_matrix handle
mkl_sparse_create(A::SparseMatrixCSC; as_CSR::Bool=false) =
    as_CSR ? mkl_sparse_create_csr(A) : mkl_sparse_create_csc(A)

function mkl_sparse_create_csr(A::SparseMatrixCSC{T}) where {T <: BlasFloat}
    h = Ref{sparse_matrix_t}()
    res = mkl_call_nolog(Val(:mkl_sparse_T_create_csr), T,
        h, SPARSE_INDEX_BASE_ONE, A.n, A.m,
        A.colptr, pointer(A.colptr, 2), A.rowval, A.nzval)
    @assert res == SPARSE_STATUS_SUCCESS
    return h[]
end

function mkl_sparse_create_csc(A::SparseMatrixCSC{T}) where {T <: BlasFloat}
    h = Ref{sparse_matrix_t}()
    res = mkl_call_nolog(Val(:mkl_sparse_T_create_csc), T,
        h, SPARSE_INDEX_BASE_ONE, A.m, A.n,
        A.colptr, pointer(A.colptr, 2), A.rowval, A.nzval)
    @assert res == SPARSE_STATUS_SUCCESS
    return h[]
end

