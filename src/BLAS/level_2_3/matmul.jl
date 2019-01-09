import Base: *, \
import LinearAlgebra: mul!, ldiv!

_get_data(A::LowerTriangular) = tril(A.data)
_get_data(A::UpperTriangular) = triu(A.data)
_get_data(A::UnitLowerTriangular) = tril(A.data)
_get_data(A::UnitUpperTriangular) = triu(A.data)
_get_data(A::Symmetric) = A.data

_unwrap_adj(x::Union{Adjoint,Transpose}) = parent(x)
_unwrap_adj(x) = x

const SparseMatrices{T} = Union{SparseMatrixCSC{T,BlasInt},
                        Symmetric{T,SparseMatrixCSC{T,BlasInt}},
                        LowerTriangular{T, SparseMatrixCSC{T,BlasInt}},
                        UnitLowerTriangular{T, SparseMatrixCSC{T,BlasInt}},
                        UpperTriangular{T, SparseMatrixCSC{T,BlasInt}},
                        UnitUpperTriangular{T, SparseMatrixCSC{T,BlasInt}}}

for T in [Complex{Float32}, Complex{Float64}, Float32, Float64]
for mat in (:StridedVector, :StridedMatrix)
for (tchar, ttype) in (('N', :()),
                       ('C', :Adjoint),
                       ('T', :Transpose))
    AT = tchar == 'N' ? :(SparseMatrixCSC{$T,BlasInt}) : :($ttype{$T,SparseMatrixCSC{$T,BlasInt}})
    @eval begin
        function mul!(α::$T, adjA::$AT,
                      B::$mat{$T}, β::$T, C::$mat{$T})
            A = _unwrap_adj(adjA)
            if isa(B, AbstractVector)
                return cscmv!($tchar, α, matdescra(A), A, B, β, C)
            else
                return cscmm!($tchar, α, matdescra(A), A, B, β, C)
            end
        end

        mul!(C::$mat{$T}, adjA::$AT, B::$mat{$T}) = mul!(one($T), adjA, B, zero($T), C)

        function (*)(adjA::$AT, B::$mat{$T})
            A = _unwrap_adj(adjA)
            if isa(B,AbstractVector)
                return mul!(zeros($T, mkl_size($tchar, A)[1]),            adjA, B)
            else
                return mul!(zeros($T, mkl_size($tchar, A)[1], size(B,2)), adjA, B)
            end
        end
    end

    for w in (:Symmetric, :LowerTriangular, :UnitLowerTriangular, :UpperTriangular, :UnitUpperTriangular)
        AT = tchar == 'N' ?
            :($w{$T,SparseMatrixCSC{$T,BlasInt}}) :
            :($ttype{$T,$w{$T,SparseMatrixCSC{$T,BlasInt}}})
        @eval begin
            function mul!(α::$T, adjA::$AT,
                         B::$mat{$T}, β::$T, C::$mat{$T})
                A = _unwrap_adj(adjA)
                if isa(B,AbstractVector)
                    return cscmv!($tchar, α, matdescra(A), _get_data(A), B, β, C)
                else
                    return cscmm!($tchar, α, matdescra(A), _get_data(A), B, β, C)
                end
            end

            mul!(C::$mat{$T}, adjA::$AT, B::$mat{$T}) = mul!(one($T), adjA, B, zero($T), C)

            function (*)(adjA::$AT, B::$mat{$T})
                A = _unwrap_adj(adjA)
                if isa(B,AbstractVector)
                    return mul!(zeros($T, mkl_size($tchar, A)[1]),            adjA, B)
                else
                    return mul!(zeros($T, mkl_size($tchar, A)[1], size(B,2)), adjA, B)
                end
            end
        end

        if w != :Symmetric
            @eval begin
                function ldiv!(α::$T, adjA::$AT,
                               B::$mat{$T}, C::$mat{$T})
                    A = _unwrap_adj(adjA)
                    if isa(B,AbstractVector)
                        return cscsv!($tchar, α, matdescra(A), _get_data(A), B, C)
                    else
                        return cscsm!($tchar, α, matdescra(A), _get_data(A), B, C)
                    end
                end

                ldiv!(C::$mat{$T}, A::$AT, B::$mat{$T}) =
                    ldiv!(one($T), A, B, C)

                function (\)(A::$AT, B::$mat{$T})
                    if isa(B,AbstractVector)
                        return ldiv!(zeros($T, size(A,1)),            A, B)
                    else
                        return ldiv!(zeros($T, size(A,1), size(B,2)), A, B)
                    end
                end
            end
        end
    end
end
end # mat
end # T
