for T in (:Float32, :Float64, :ComplexF32, :ComplexF64)
  INT_TYPES = Base.USE_BLAS64 ? (:Int32, :Int64) : (:Int32,)
  for INT in INT_TYPES
    for SparseMatrix in (:(SparseMatrixCSC{$T,$INT}), :(MKLSparse.SparseMatrixCSR{$T,$INT}), :(MKLSparse.SparseMatrixCOO{$T,$INT}))

      fname_mv   = Symbol("mkl_sparse_", mkl_type_specifier(T), "_mv"  , mkl_integer_specifier(INT))
      fname_mm   = Symbol("mkl_sparse_", mkl_type_specifier(T), "_mm"  , mkl_integer_specifier(INT))
      fname_trsv = Symbol("mkl_sparse_", mkl_type_specifier(T), "_trsv", mkl_integer_specifier(INT))
      fname_trsm = Symbol("mkl_sparse_", mkl_type_specifier(T), "_trsm", mkl_integer_specifier(INT))

      @eval begin
        function mv!(operation::Char, alpha::$T, A::$SparseMatrix, descr::matrix_descr, x::StridedVector{$T}, beta::$T, y::StridedVector{$T})
          _check_transa(operation)
          _check_mat_mult_matvec(y, A, x, operation)
          __counter[] += 1
          $fname_mv(operation, alpha, MKLSparseMatrix(A), descr, x, beta, y)
          return y
        end

        function mm!(operation::Char, alpha::$T, A::$SparseMatrix, descr::matrix_descr, x::StridedMatrix{$T}, beta::$T, y::StridedMatrix{$T})
          _check_transa(operation)
          _check_mat_mult_matvec(y, A, x, operation)
          __counter[] += 1
          columns = size(y, 2)
          ldx = stride(x, 2)
          ldy = stride(y, 2)
          $fname_mm(operation, alpha, MKLSparseMatrix(A), descr, 'C', x, columns, ldx, beta, y, ldy)
          return y
        end

        function trsv!(operation::Char, alpha::$T, A::$SparseMatrix, descr::matrix_descr, x::StridedVector{$T}, y::StridedVector{$T})
          checksquare(A)
          _check_transa(operation)
          _check_mat_mult_matvec(y, A, x, operation)
          __counter[] += 1
          $fname_trsv(operation, alpha, MKLSparseMatrix(A), descr, x, y)
          return y
        end

        function trsm!(operation::Char, alpha::$T, A::$SparseMatrix, descr::matrix_descr, x::StridedMatrix{$T}, y::StridedMatrix{$T})
          checksquare(A)
          _check_transa(operation)
          _check_mat_mult_matvec(y, A, x, operation)
          __counter[] += 1
          columns = size(y, 2)
          ldx = stride(x, 2)
          ldy = stride(y, 2)
          $fname_trsm(operation, alpha, MKLSparseMatrix(A), descr, 'C', x, columns, ldx, y, ldy)
          return y
        end
      end
    end
  end
end
