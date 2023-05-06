_data(A::AbstractMatrix) = A
_data(A::LowerTriangular) = tril(A.data)
_data(A::UpperTriangular) = triu(A.data)
_data(A::UnitLowerTriangular) = tril(A.data)
_data(A::UnitUpperTriangular) = triu(A.data)
_data(A::Symmetric) = A.data

# returns a tuple of matdescra and unwrapped A
describe_and_unwrap(A::AbstractMatrix) = (matdescra(A), _data(A)) # unwrap adjoint/transpose
describe_and_unwrap(A::Union{Adjoint,Transpose}) = describe_and_unwrap(parent(A)) # unwrap adjoint/transpose

const SparseMatrices{T} = Union{SparseMatrixCSC{T,BlasInt},
                        Symmetric{T,SparseMatrixCSC{T,BlasInt}},
                        LowerTriangular{T, SparseMatrixCSC{T,BlasInt}},
                        UnitLowerTriangular{T, SparseMatrixCSC{T,BlasInt}},
                        UpperTriangular{T, SparseMatrixCSC{T,BlasInt}},
                        UnitUpperTriangular{T, SparseMatrixCSC{T,BlasInt}}}

# (re)define Base and LinearAlgebra methods to use MKLSparse implementations where appropriate
for T in (Complex{Float32}, Complex{Float64}, Float32, Float64),
    ttype in (nothing, :Adjoint, :Transpose)

    tchar = mkl_operation_code(ttype)
    # mul!(C, A, B, a, b), where A is SparseMatrixCSC or adjoint/transpose of it
    AT = isnothing(ttype) ? :(SparseMatrixCSC{$T,BlasInt}) : :($ttype{$T,SparseMatrixCSC{$T,BlasInt}})
    @eval begin
        LinearAlgebra.mul!(C::StridedVector{$T}, A::$AT, B::StridedVector{$T}, α::Number, β::Number) =
            cscmv!($tchar, $T(α), describe_and_unwrap(A)..., B, $T(β), C)

        LinearAlgebra.mul!(C::StridedMatrix{$T}, A::$AT, B::StridedMatrix{$T}, α::Number, β::Number) =
            cscmm!($tchar, $T(α), describe_and_unwrap(A)..., B, $T(β), C)
    end

    for w in (:Symmetric, :LowerTriangular, :UnitLowerTriangular, :UpperTriangular, :UnitUpperTriangular)
        AT = isnothing(ttype) ?
            :($w{$T,SparseMatrixCSC{$T,BlasInt}}) :
            :($ttype{$T,$w{$T,SparseMatrixCSC{$T,BlasInt}}})
        BT = :(Union{StridedMatrix{$T}, StridedVector{$T}})

        # mul!(C, A, B, a, b), where A is Symmetric/LowTri etc of a SparseMatrixCSC or adjoint/transpose of it
        # it has special implementation in Base, so we redefine it to use MKLSparse implementation
        @eval begin
            LinearAlgebra.mul!(C::StridedVector{$T}, A::$AT, B::StridedVector{$T}, α::Number, β::Number) =
                cscmv!($tchar, $T(α), describe_and_unwrap(A)..., B, $T(β), C)

            LinearAlgebra.mul!(C::StridedMatrix{$T}, A::$AT, B::StridedMatrix{$T}, α::Number, β::Number) =
                cscmm!($tchar, $T(α), describe_and_unwrap(A)..., B, $T(β), C)

            LinearAlgebra.mul!(C::BT, A::$AT, B::BT) where BT <: $BT = mul!(C, A, B, one($T), zero($T))

            # base A*B converts A to dense, so we redefine it to use MKLSparse-enabled mul!
            Base.:(*)(A::$AT, B::StridedVector{$T}) = mul!(Vector{$T}(undef, size(A, 1)), A, B)
            Base.:(*)(A::$AT, B::StridedMatrix{$T}) = mul!(Matrix{$T}(undef, size(A, 1), size(B, 2)), A, B)
        end

        # define 4-arg ldiv!(C, A, B, a) (C := alpha*inv(A)*B) that is not present in standard LinearAlgrebra,
        # redefine 3-arg ldiv!(C, A, B) using 4-arg ldiv!(C, A, B, 1)
        # here A is LowerTri/UpperTri etc of a SparseMatrixCSC or adjoint/transpose of it (Symmetric not supported)
        if w != :Symmetric
            @eval begin
                LinearAlgebra.ldiv!(C::StridedVector{$T}, A::$AT, B::StridedVector{$T}, α::Number) =
                    cscsv!($tchar, $T(α), describe_and_unwrap(A)..., B, C)
                LinearAlgebra.ldiv!(C::StridedMatrix{$T}, A::$AT, B::StridedMatrix{$T}, α::Number) =
                    cscsm!($tchar, $T(α), describe_and_unwrap(A)..., B, C)

                LinearAlgebra.ldiv!(C::BT, A::$AT, B::BT) where BT <: $BT = ldiv!(C, A, B, one($T))

                # base A\B converts A to dense, so we redefine it to use MKLSparse-enabled ldiv!
                Base.:(\)(A::$AT, B::StridedVector{$T}) = ldiv!(Vector{$T}(undef, size(A, 1)), A, B)
                Base.:(\)(A::$AT, B::StridedMatrix{$T}) = ldiv!(Matrix{$T}(undef, size(A, 1), size(B, 2)), A, B)
            end
        end
    end
end # T, ttype
