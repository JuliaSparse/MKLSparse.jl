import Base: \, *
import LinearAlgebra: mul!, ldiv!

MKLSparseMat{T} = Union{SparseArrays.AbstractSparseMatrixCSC{T}, SparseMatrixCSR{T}, SparseMatrixCOO{T}}

SimpleOrSpecialMat{T, M} = Union{M, LowerTriangular{T,<:M}, UpperTriangular{T,<:M},
                                 UnitLowerTriangular{T,<:M}, UnitUpperTriangular{T,<:M},
                                 Symmetric{T,<:M}, Hermitian{T,<:M}}
SimpleOrSpecialOrAdjMat{T, M} = Union{SimpleOrSpecialMat{T, M},
                                      Adjoint{T, <:SimpleOrSpecialMat{T, M}},
                                      Transpose{T, <:SimpleOrSpecialMat{T, M}}}

unwrapa(A::AbstractMatrix) = A
unwrapa(A::Union{LowerTriangular, UpperTriangular,
                 UnitLowerTriangular, UnitUpperTriangular,
                 Symmetric, Hermitian}) = parent(A)

# returns a tuple of transa, matdescra and unwrapped A
describe_and_unwrap(A::AbstractMatrix) = ('N', matrixdescra(A), unwrapa(A))
describe_and_unwrap(A::Adjoint) = ('C', matrixdescra(A), unwrapa(parent(A)))
describe_and_unwrap(A::Transpose) = ('T', matrixdescra(A), unwrapa(parent(A)))

# 5-arg mul!()
function mul!(y::StridedVector{T}, A::SimpleOrSpecialOrAdjMat{T, S}, x::StridedVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    transA, descrA, unwrapA = describe_and_unwrap(A)
    mv!(transA, T(alpha), unwrapA, descrA, x, T(beta), y)
end

function mul!(C::StridedMatrix{T}, A::SimpleOrSpecialOrAdjMat{T, S}, B::StridedMatrix{T}, alpha::Number, beta::Number) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    transA, descrA, unwrapA = describe_and_unwrap(A)
    mm!(transA, T(alpha), unwrapA, descrA, B, T(beta), C)
end

# define 4-arg ldiv!(C, A, B, a) (C := alpha*inv(A)*B) that is not present in standard LinearAlgrebra
# redefine 3-arg ldiv!(C, A, B) using 4-arg ldiv!(C, A, B, 1)
function ldiv!(y::StridedVector{T}, A::SimpleOrSpecialOrAdjMat{T, S}, x::StridedVector{T}, alpha::Number = one(T)) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    transA, descrA, unwrapA = describe_and_unwrap(A)
    trsv!(transA, alpha, unwrapA, descrA, x, y)
end

function LinearAlgebra.ldiv!(C::StridedMatrix{T}, A::SimpleOrSpecialOrAdjMat{T, S}, B::StridedMatrix{T}, alpha::Number = one(T)) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    transA, descrA, unwrapA = describe_and_unwrap(A)
    trsm!(transA, alpha, unwrapA, descrA, B, C)
end

function (*)(A::SimpleOrSpecialOrAdjMat{T, S}, x::StridedVector{T}) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    m, n = size(A)
    y = Vector{T}(undef, m)
    return mul!(y, A, x, one(T), zero(T))
end

function (*)(A::SimpleOrSpecialOrAdjMat{T, S}, B::StridedMatrix{T}) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    m, k = size(A)
    p, n = size(B)
    C = Matrix{T}(undef, m, n)
    return mul!(C, A, B, one(T), zero(T))
end

function (\)(A::SimpleOrSpecialOrAdjMat{T, S}, x::StridedVector{T}) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    n = length(x)
    y = Vector{T}(undef, n)
    return ldiv!(y, A, x)
end

function (\)(A::SimpleOrSpecialOrAdjMat{T, S}, B::StridedMatrix{T}) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    m, n = size(B)
    C = Matrix{T}(undef, m, n)
    return ldiv!(C, A, B)
end
