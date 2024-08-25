import Base: \, *
import LinearAlgebra: mul!, ldiv!

MKLSparseMat{T} = Union{SparseArrays.AbstractSparseMatrixCSC{T}, SparseMatrixCSR{T}, SparseMatrixCOO{T}}

SimpleOrAdjMat{T, M} = Union{M, Adjoint{T, <:M}, Transpose{T, <:M}}

SpecialMat{T, M} = Union{LowerTriangular{T,<:M}, UpperTriangular{T,<:M},
                         UnitLowerTriangular{T,<:M}, UnitUpperTriangular{T,<:M},
                         Symmetric{T,<:M}, Hermitian{T,<:M}}
SimpleOrSpecialMat{T, M} = Union{M, SpecialMat{T, <:M}}
SimpleOrSpecialOrAdjMat{T, M} = Union{SimpleOrAdjMat{T, <:SimpleOrSpecialMat{T, <:M}},
                                      SimpleOrSpecialMat{T, <:SimpleOrAdjMat{T, <:M}}}

unwrapa(A::AbstractMatrix) = A
unwrapa(A::Union{Adjoint, Transpose}) = unwrapa(parent(A))
unwrapa(A::SpecialMat) = unwrapa(parent(A))

# returns a tuple of transa, matdescra and unwrapped A
describe_and_unwrap(A::AbstractMatrix) = ('N', matrix_descr(A), unwrapa(A))
describe_and_unwrap(A::Adjoint) = ('C', matrix_descr(A), unwrapa(parent(A)))
describe_and_unwrap(A::Transpose) = ('T', matrix_descr(A), unwrapa(parent(A)))
describe_and_unwrap(A::LowerTriangular{<:Any, T}) where T <: Union{Adjoint, Transpose} =
    (T <: Adjoint ? 'C' : 'T', matrix_descr('T', 'U', 'N'), unwrapa(A))
describe_and_unwrap(A::UpperTriangular{<:Any, T}) where T <: Union{Adjoint, Transpose} =
    (T <: Adjoint ? 'C' : 'T', matrix_descr('T', 'L', 'N'), unwrapa(A))
describe_and_unwrap(A::UnitLowerTriangular{<:Any, T}) where T <: Union{Adjoint, Transpose} =
    (T <: Adjoint ? 'C' : 'T', matrix_descr('T', 'U', 'U'), unwrapa(A))
describe_and_unwrap(A::UnitUpperTriangular{<:Any, T}) where T <: Union{Adjoint, Transpose} =
    (T <: Adjoint ? 'C' : 'T', matrix_descr('T', 'L', 'U'), unwrapa(A))
describe_and_unwrap(A::Symmetric{<:Any, T}) where T <: Union{Adjoint, Transpose} =
    (T <: Transpose || (eltype(A) <: Real) ? 'N' : 'C', matrix_descr('S', A.uplo, 'N'), unwrapa(A))
describe_and_unwrap(A::Hermitian{<:Any, T}) where T <: Union{Adjoint, Transpose} =
    (T <: Adjoint || (eltype(A) <: Real) ? 'N' : 'T', matrix_descr('H', A.uplo, 'N'), unwrapa(A))

# 5-arg mul!()
function mul!(y::StridedVector{T}, A::SimpleOrSpecialOrAdjMat{T, S},
              x::StridedVector{T}, alpha::Number, beta::Number
) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    transA, descrA, unwrapA = describe_and_unwrap(A)
    mv!(transA, T(alpha), unwrapA, descrA, x, T(beta), y)
end

function mul!(C::StridedMatrix{T}, A::SimpleOrSpecialOrAdjMat{T, S},
              B::StridedMatrix{T}, alpha::Number, beta::Number
) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    transA, descrA, unwrapA = describe_and_unwrap(A)
    mm!(transA, T(alpha), unwrapA, descrA, B, T(beta), C)
end

# ColMajorRes = ColMajorMtx*SparseMatrixCSC is implemented via
# RowMajorRes = SparseMatrixCSR*RowMajorMtx Sparse MKL BLAS calls
# Switching the B layout from CSC to CSR is required, because MKLSparse
# does not support CSC 1-based multiplication with row-major matrices.
# Only CSC is supported as for the other sparse formats the combination
# of indexing, storage and dense layout would be unsupported,
# see https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/mkl-sparse-mm.html
# (one potential workaround is to temporarily switch to 0-based indexing)
function mul!(C::StridedMatrix{T}, A::StridedMatrix{T},
              B::SimpleOrSpecialOrAdjMat{T, S}, alpha::Number, beta::Number
) where {T <: BlasFloat, S <: SparseArrays.AbstractSparseMatrixCSC{T}}
    transB, descrB, unwrapB = describe_and_unwrap(B)
    mm!(transB, T(alpha), lazypermutedims(unwrapB), lazypermutedims(descrB), A,
        T(beta), C, dense_layout = SPARSE_LAYOUT_ROW_MAJOR)
end

# 3-arg mul!() calls 5-arg mul!()
mul!(y::StridedVector{T}, A::SimpleOrSpecialOrAdjMat{T, S},
     x::StridedVector{T}) where {T <: BlasFloat, S <: MKLSparseMat{T}} =
    mul!(y, A, x, one(T), zero(T))
mul!(C::StridedMatrix{T}, A::SimpleOrSpecialOrAdjMat{T, S},
     B::StridedMatrix{T}) where {T <: BlasFloat, S <: MKLSparseMat{T}} =
    mul!(C, A, B, one(T), zero(T))
mul!(C::StridedMatrix{T}, A::StridedMatrix{T},
     B::SimpleOrSpecialOrAdjMat{T, S}) where {T <: BlasFloat, S <: MKLSparseMat{T}} =
    mul!(C, A, B, one(T), zero(T))

# define 4-arg ldiv!(C, A, B, a) (C := alpha*inv(A)*B) that is not present in standard LinearAlgrebra
# redefine 3-arg ldiv!(C, A, B) using 4-arg ldiv!(C, A, B, 1)
function ldiv!(y::StridedVector{T}, A::SimpleOrSpecialOrAdjMat{T, S},
               x::StridedVector{T}, alpha::Number = one(T)) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    transA, descrA, unwrapA = describe_and_unwrap(A)
    trsv!(transA, alpha, unwrapA, descrA, x, y)
end

function LinearAlgebra.ldiv!(C::StridedMatrix{T}, A::SimpleOrSpecialOrAdjMat{T, S},
                             B::StridedMatrix{T}, alpha::Number = one(T)) where {T <: BlasFloat, S <: MKLSparseMat{T}}
    transA, descrA, unwrapA = describe_and_unwrap(A)
    trsm!(transA, alpha, unwrapA, descrA, B, C)
end

if VERSION < v"1.10"
# stdlib v1.9 does not provide these methods

(*)(A::SimpleOrSpecialOrAdjMat{T, S}, x::StridedVector{T}) where {T <: BlasFloat, S <: MKLSparseMat{T}} =
    mul!(Vector{T}(undef, size(A, 1)), A, x)

(*)(A::SimpleOrSpecialOrAdjMat{T, S}, B::StridedMatrix{T}) where {T <: BlasFloat, S <: MKLSparseMat{T}} =
    mul!(Matrix{T}(undef, size(A, 1), size(B, 2)), A, B)

end # if VERSION < v"1.10"

(*)(A::StridedMatrix{T}, B::SimpleOrSpecialOrAdjMat{T, S}) where {T <: BlasFloat, S <: MKLSparseMat{T}} =
    mul!(Matrix{T}(undef, size(A, 1), size(B, 2)), A, B)

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
