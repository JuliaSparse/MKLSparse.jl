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

Base.size(A::MKLSparse.SparseMatrixCOO) = (A.m, A.n)
Base.size(A::MKLSparse.SparseMatrixCSR) = (A.m, A.n)

SparseArrays.nnz(A::MKLSparse.SparseMatrixCOO) = length(A.vals)
SparseArrays.nnz(A::MKLSparse.SparseMatrixCSR) = length(A.nzval)

matrixdescra(A::MKLSparse.SparseMatrixCSR) = matrix_descr('G', 'F', 'N')
matrixdescra(A::MKLSparse.SparseMatrixCOO) = matrix_descr('G', 'F', 'N')

mutable struct MKLSparseMatrix
  handle::sparse_matrix_t
end

Base.unsafe_convert(::Type{sparse_matrix_t}, desc::MKLSparseMatrix) = desc.handle

for T in (:Float32, :Float64, :ComplexF32, :ComplexF64)
  INT_TYPES = Base.USE_BLAS64 ? (:Int32, :Int64) : (:Int32,)
  for INT in INT_TYPES

    create_coo = Symbol("mkl_sparse_", mkl_type_specifier(T), "_create_coo", mkl_integer_specifier(INT))
    create_csc = Symbol("mkl_sparse_", mkl_type_specifier(T), "_create_csc", mkl_integer_specifier(INT))
    create_csr = Symbol("mkl_sparse_", mkl_type_specifier(T), "_create_csr", mkl_integer_specifier(INT))
    sparse_destroy = (INT == :Int32) ? :mkl_sparse_destroy : :mkl_sparse_destroy_64

    @eval begin
      # SparseMatrixCOO
      function MKLSparseMatrix(A::MKLSparse.SparseMatrixCOO{$T, $INT}, IndexBase::Char='O')
        descr_ref = Ref{sparse_matrix_t}()
        $create_coo(descr_ref, IndexBase, A.m, A.n, nnz(A), A.rows, A.cols, A.vals)
        obj = MKLSparseMatrix(descr_ref[])
        finalizer($sparse_destroy, obj)
        return obj
      end

      # SparseMatrixCSR
      function MKLSparseMatrix(A::MKLSparse.SparseMatrixCSR{$T, $INT}, IndexBase::Char='O')
        descr_ref = Ref{sparse_matrix_t}()
        $create_csr(descr_ref, IndexBase, A.m, A.n, A.rowptr, pointer(A.rowptr, 2), A.colval, A.nzval)
        obj = MKLSparseMatrix(descr_ref[])
        finalizer($sparse_destroy, obj)
        return obj
      end

      # SparseMatrixCSC
      function MKLSparseMatrix(A::SparseMatrixCSC{$T, $INT}, IndexBase::Char='O')
        descr_ref = Ref{sparse_matrix_t}()
        $create_csc(descr_ref, IndexBase, A.m, A.n, A.colptr, pointer(A.colptr, 2), A.rowval, A.nzval)
        obj = MKLSparseMatrix(descr_ref[])
        finalizer($sparse_destroy, obj)
        return obj
      end
    end
  end
end
