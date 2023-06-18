# SparseBLAS description of A structure
# see https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/sparse-blas-level-2-and-level-3-routines/interface-consideration.html#interface-consideration_GUID-0C896CB1-25B6-4B53-A425-C5954B4C22F9
matdescra(A::LowerTriangular) = "TLNF"
matdescra(A::UpperTriangular) = "TUNF"
matdescra(A::Diagonal) = "DUNF"
matdescra(A::UnitLowerTriangular) = "TLUF"
matdescra(A::UnitUpperTriangular) = "TUUF"
matdescra(A::Symmetric) = string('S', A.uplo, 'N', 'F')
matdescra(A::Hermitian) = string('H', A.uplo, 'N', 'F')
matdescra(A::SparseMatrixCSC) = "GFNF"
