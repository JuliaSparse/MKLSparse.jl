# The increments to the `__counter` variable is for testing purposes

function _check_transa(t::Char)
    if !(t in ('C', 'N', 'T'))
        error("transa: is '$t', must be 'N', 'T', or 'C'")
    end
end

mkl_size(t::Char, M::AbstractVecOrMat) = t == 'N' ? size(M) : reverse(size(M))


# Checks sizes for the multiplication C <- A * B
function _check_mat_mult_matvec(C, A, B, tA)
    _size(v::AbstractMatrix) = size(v)
    _size(v::AbstractVector) = (size(v,1), 1)
    _str(v::AbstractMatrix) = string("[", size(v,1), ", ", size(v,2), "]")
    _str(v::AbstractVector) = string("[", size(v,1), "]")
    mA, nA = mkl_size(tA, A)
    mB, nB = _size(B)
    mC, nC = _size(C)
    if nA != mB || mC != mA || nC != nB
        t = ""
        if tA == 'T'; t = ".\'"; end
        if tA == 'C'; t = "\'"; end
        str = string("arrays had inconsistent dimensions for C <- A", t, " * B: ", _str(C), " <- ", _str(A), t, " * ", _str(B))
        throw(DimensionMismatch(str))
    end
end

for (mv, sv, mm, sm, T) in ((:mkl_scscmv, :mkl_scscsv, :mkl_scscmm, :mkl_scscsm, :Float32),
                            (:mkl_dcscmv, :mkl_dcscsv, :mkl_dcscmm, :mkl_dcscsm, :Float64),
                            (:mkl_ccscmv, :mkl_ccscsv, :mkl_ccscmm, :mkl_ccscsm, :ComplexF32),
                            (:mkl_zcscmv, :mkl_zcscsv, :mkl_zcscmm, :mkl_zcscsm, :ComplexF64))
@eval begin
function cscmv!(transa::Char, α::$T, matdescra::String,
                A::SparseMatrixCSC{$T, BlasInt}, x::StridedVector{$T},
                β::$T, y::StridedVector{$T})
    _check_transa(transa)
    _check_mat_mult_matvec(y, A, x, transa)
    __counter[] += 1
    ccall(($(string(mv)), :libmkl_rt), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$T},
         Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt},
         Ptr{BlasInt}, Ptr{$T}, Ref{$T}, Ptr{$T}),
        transa, A.m, A.n, α,
        matdescra, A.nzval, A.rowval, A.colptr,
        pointer(A.colptr, 2), x, β, y)
    return y
end

function cscmm!(transa::Char, α::$T, matdescra::String,
                A::SparseMatrixCSC{$T, BlasInt}, B::StridedMatrix{$T},
                β::$T, C::StridedMatrix{$T})
    _check_transa(transa)
    _check_mat_mult_matvec(C, A, B, transa)
    mB, nB = size(B)
    mC, nC = size(C)
    __counter[] += 1
    ccall(($(string(mm)), :libmkl_rt), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
         Ref{$T}, Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt},
         Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ref{BlasInt},
         Ref{$T}, Ptr{$T}, Ref{BlasInt}),
        transa, A.m, nC, A.n,
        α, matdescra, A.nzval, A.rowval,
        A.colptr, pointer(A.colptr, 2), B, mB,
        β, C, mC)
    return C
end

function cscsv!(transa::Char, α::$T, matdescra::String,
                A::SparseMatrixCSC{$T, BlasInt}, x::StridedVector{$T},
                y::StridedVector{$T})
    n = checksquare(A)
    _check_transa(transa)
    _check_mat_mult_matvec(y, A, x, transa)
    __counter[] += 1
    ccall(($(string(sv)), :libmkl_rt), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{$T}, Ptr{UInt8},
         Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
         Ptr{$T}, Ptr{$T}),
        transa, A.m, α, matdescra,
        A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2),
        x, y)
    return y
end

function cscsm!(transa::Char, α::$T, matdescra::String,
                A::SparseMatrixCSC{$T, BlasInt}, B::StridedMatrix{$T},
                C::StridedMatrix{$T})
    mB, nB = size(B)
    mC, nC = size(C)
    n = checksquare(A)
    _check_transa(transa)
    _check_mat_mult_matvec(C, A, B, transa)
    __counter[] += 1
    ccall(($(string(sm)), :libmkl_rt), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$T},
         Ptr{UInt8}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt},
         Ptr{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T},
         Ref{BlasInt}),
        transa, A.n, nC, α,
        matdescra, A.nzval, A.rowval, A.colptr,
        pointer(A.colptr, 2), B, mB, C,
        mC)
    return C
end

end # @eval
end
