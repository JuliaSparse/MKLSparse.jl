function Base.LinAlg.A_mul_B!{T<:BlasFloat}(α::T, A::SparseMatrixCSC{T,BlasInt},
                                            B::StridedVecOrMat{T}, β::T, C::StridedVecOrMat{T})
    A.n == size(B, 1) || throw(DimensionMismatch())
    A.m == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    isa(B,AbstractVector) ?
        cscmv!('N',α,matdescra(A),A,B,β,C) :
        cscmm!('N',α,matdescra(A),A,B,β,C)
end

function Base.LinAlg.Ac_mul_B!{T<:BlasFloat}(α::T, A::SparseMatrixCSC{T,BlasInt},
                                             B::StridedVecOrMat{T}, β::T, C::StridedVecOrMat{T})
    A.n == size(C, 1) || throw(DimensionMismatch())
    A.m == size(B, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    isa(B,AbstractVector) ?
        cscmv!('T',α,matdescra(A),A,B,β,C) :
        cscmm!('T',α,matdescra(A),A,B,β,C)
end

for w in (:LowerTriangular,:UnitLowerTriangular,:UpperTriangular,:UnitUpperTriangular,:Symmetric)
    @eval begin
        function Base.LinAlg.A_mul_B!{T<:BlasFloat}(α::T, A::$w{T,SparseMatrixCSC{T,BlasInt}},
                                             B::StridedVecOrMat{T}, β::T, C::StridedVecOrMat{T})
            n = size(A,2)
            n == size(C, 1) == size(B,1) || throw(DimensionMismatch())
            size(B, 2) == size(C, 2) || throw(DimensionMismatch())
            isa(B,AbstractVector) ?
                cscmv!('N',α,matdescra(A),A.data,B,β,C) :
                cscmm!('N',α,matdescra(A),A.data,B,β,C)
        end
        function Base.LinAlg.Ac_mul_B!{T<:BlasFloat}(α::T, A::$w{T,SparseMatrixCSC{T,BlasInt}},
                                                     B::StridedVecOrMat{T}, β::T, C::StridedVecOrMat{T})
            n = size(A,2)
            n == size(C, 1) == size(B, 1) || throw(DimensionMismatch())
            size(B, 2) == size(C, 2) || throw(DimensionMismatch())
            isa(B,AbstractVector) ?
                cscmv!('T',α,matdescra(A),A.data,B,β,C) :
                cscmm!('T',α,matdescra(A),A.data,B,β,C)
        end
    end
end

for w in (:LowerTriangular,:UnitLowerTriangular,:UpperTriangular,:UnitUpperTriangular)
    @eval begin
        function Base.LinAlg.A_ldiv_B!{T<:BlasFloat}(α::T, A::$w{T,SparseMatrixCSC{T,BlasInt}},
                                                     B::StridedVecOrMat{T}, C::StridedVecOrMat{T})
            @show typeof(A)
            size(A,2) == size(B,1) == size(C,1) || throw(DimensionMismatch())
            size(B,2) == size(C,2) || throw(DimensionMismatch())
            isa(B,AbstractVector) ?
                cscsv!('N',α,matdescra(A),A.data,B,C) :
                cscsm!('N',α,matdescra(A),A.data,B,C)
        end
        function Base.LinAlg.Ac_ldiv_B!{T<:BlasFloat}(α::T, A::$w{T,SparseMatrixCSC{T,BlasInt}},
                                                      B::StridedVecOrMat{T}, C::StridedVecOrMat{T})
            @show typeof(A)
            size(A,2) == size(C,1) == size(B,1) || throw(DimensionMismatch())
            size(B,2) == size(C,2) || throw(DimensionMismatch())
            isa(B,AbstractVector) ?
                cscsv!('T',α,matdescra(A),A.data,B,C) :
                cscsm!('T',α,matdescra(A),A.data,B,C)
        end
    end
end
