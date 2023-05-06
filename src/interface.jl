import Base: \, *
import LinearAlgebra: mul!, ldiv!

for T in (:Float32, :Float64, :ComplexF32, :ComplexF64)
  INT_TYPES = Base.USE_BLAS64 ? (:Int32, :Int64) : (:Int32,)
  for INT in INT_TYPES

    tag_wrappers = ((identity                 , identity          ),
                    (M -> :(Symmetric{$T, $M}), A -> :(parent($A))),
                    (M -> :(Hermitian{$T, $M}), A -> :(parent($A))))

    triangle_wrappers = ((M -> :(LowerTriangular{$T, $M})    , A -> :(parent($A))),
                         (M -> :(UnitLowerTriangular{$T, $M}), A -> :(parent($A))),
                         (M -> :(UpperTriangular{$T, $M})    , A -> :(parent($A))),
                         (M -> :(UnitUpperTriangular{$T, $M}), A -> :(parent($A))))

    op_wrappers = ((identity                 , 'N', identity          ),
                   (M -> :(Transpose{$T, $M}), 'T', A -> :(parent($A))),
                   (M -> :(Adjoint{$T, $M})  , 'C', A -> :(parent($A))))

    for SparseMatrixType in (:(SparseMatrixCSC{$T, $INT}), :(MKLSparse.SparseMatrixCOO{$T, $INT}), :(MKLSparse.SparseMatrixCSR{$T, $INT}))
      for (taga, untaga) in tag_wrappers, (wrapa, transa, unwrapa) in op_wrappers
        TypeA = wrapa(taga(SparseMatrixType))

        @eval begin
          function LinearAlgebra.mul!(y::StridedVector{$T}, A::$TypeA, x::StridedVector{$T}, alpha::Number, beta::Number)
              # return cscmv!($transa, $T(alpha), $matdescra(A), $(untaga(unwrapa(:A))), x, $T(beta), y)
              return mv!($transa, $T(alpha), $(untaga(unwrapa(:A))), $matrixdescra(A), x, $T(beta), y)
          end

          function LinearAlgebra.mul!(C::StridedMatrix{$T}, A::$TypeA, B::StridedMatrix{$T}, alpha::Number, beta::Number)
              # return cscmm!($transa, $T(alpha), $matdescra(A), $(untaga(unwrapa(:A))), B, $T(beta), C)
              return mm!($transa, $T(alpha), $(untaga(unwrapa(:A))), $matrixdescra(A), B, $T(beta), C)
          end
        end
      end

      for (trianglea, untrianglea) in triangle_wrappers, (wrapa, transa, unwrapa) in op_wrappers
        TypeA = wrapa(trianglea(SparseMatrixType))

        @eval begin
          function LinearAlgebra.mul!(y::StridedVector{$T}, A::$TypeA, x::StridedVector{$T}, alpha::Number, beta::Number)
              # return cscmv!($transa, $T(alpha), $matdescra(A), $(untrianglea(unwrapa(:A))), x, $T(beta), y)
              return mv!($transa, $T(alpha), $(untrianglea(unwrapa(:A))), $matrixdescra(A), x, $T(beta), y)
          end

          function LinearAlgebra.mul!(C::StridedMatrix{$T}, A::$TypeA, B::StridedMatrix{$T}, alpha::Number, beta::Number)
              # return cscmm!($transa, $T(alpha), $matdescra(A), $(untrianglea(unwrapa(:A))), B, $T(beta), C)
              return mm!($transa, $T(alpha), $(untrianglea(unwrapa(:A))), $matrixdescra(A), B, $T(beta), C)
          end

          # define 4-arg ldiv!(C, A, B, a) (C := alpha*inv(A)*B) that is not present in standard LinearAlgrebra
          # redefine 3-arg ldiv!(C, A, B) using 4-arg ldiv!(C, A, B, 1)
          function LinearAlgebra.ldiv!(y::StridedVector{$T}, A::$TypeA, x::StridedVector{$T}, alpha::Number = one($T))
            # return cscsv!($transa, alpha, $matdescra(A), $(untrianglea(unwrapa(:A))), x, y)
            return trsv!($transa, alpha, $(untrianglea(unwrapa(:A))), $matrixdescra(A), x, y)
          end

          function LinearAlgebra.ldiv!(C::StridedMatrix{$T}, A::$TypeA, B::StridedMatrix{$T}, alpha::Number = one($T))
            # return cscsm!($transa, alpha, $matdescra(A), $(untrianglea(unwrapa(:A))), B, C)
            return trsm!($transa, alpha, $(untrianglea(unwrapa(:A))), $matrixdescra(A), B, C)
          end

          function (*)(A::$TypeA, x::StridedVector{$T})
            m, n = size(A)
            y = Vector{$T}(undef, m)
            return mul!(y, A, x, one($T), zero($T))
          end

          function (*)(A::$TypeA, B::StridedMatrix{$T})
            m, k = size(A)
            p, n = size(B)
            C = Matrix{$T}(undef, m, n)
            return mul!(C, A, B, one($T), zero($T))
          end

          function (\)(A::$TypeA, x::StridedVector{$T})
            n = length(x)
            y = Vector{$T}(undef, n)
            return ldiv!(y, A, x)
          end

          function (\)(A::$TypeA, B::StridedMatrix{$T})
            m, n = size(B)
            C = Matrix{$T}(undef, m, n)
            return ldiv!(C, A, B)
          end
        end
      end
    end
  end
end
