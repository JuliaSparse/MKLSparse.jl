import Base.LinAlg: BlasFloat,BlasInt,UnitLowerTriangular,UnitUpperTriangular

matdescra{T<:BlasFloat}(A::LowerTriangular{T,SparseMatrixCSC{T,BlasInt}}) = "TLNF"
matdescra{T<:BlasFloat}(A::UpperTriangular{T,SparseMatrixCSC{T,BlasInt}}) = "TUNF"
matdescra{T<:BlasFloat}(A::UnitLowerTriangular{T,SparseMatrixCSC{T,BlasInt}}) = "TLUF"
matdescra{T<:BlasFloat}(A::UnitUpperTriangular{T,SparseMatrixCSC{T,BlasInt}}) = "TUUF"
matdescra{T<:BlasFloat}(A::Symmetric{T,SparseMatrixCSC{T,BlasInt}}) = ASCIIString(string('S',A.uplo,'N','F'))
matdescra{T<:BlasFloat}(A::Hermitian{T,SparseMatrixCSC{T,BlasInt}}) = ASCIIString(string('H',A.uplo,'N','F'))
matdescra{T<:BlasFloat}(A::SparseMatrixCSC{T,BlasInt}) = "GUUF"
