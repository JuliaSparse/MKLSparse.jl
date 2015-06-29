srand(1234)

import Base.LinAlg: At_ldiv_B!, Ac_ldiv_B!
import MKLSparse.DSS: MatrixSymStructure
for T in (Float32, Float64, Complex64, Complex128)
    n = 5
    A1 = sparse(rand(T, n, n))
    A2 = A1 + A1'
    A3 = A1 + A1.'
    A4 = A1'A1
    B = rand(T, n, n)
    X = similar(B)

    @test_throws DimensionMismatch A_ldiv_B!(A1, rand(T, n, n+1), X)
    @test_throws DimensionMismatch A_ldiv_B!(A1, B, rand(T, n, n+1))
    @test_throws DimensionMismatch A_ldiv_B!(sparse(rand(T, n+1, n+1)), B, X)

    for A in SparseMatrixCSC[A1, A2, A3, A4]
        if T == Float32 || T == Complex64
            continue
        end
        msm = MatrixSymStructure(A)
        @test issym(A) == issym(msm)
        @test ishermitian(A) == ishermitian(msm)
    end

    for A in SparseMatrixCSC[A1, A2, A3, A4]
        @test_approx_eq A_ldiv_B!(A, B, X) full(A)\B
        @test_approx_eq At_ldiv_B!(A, B, X) full(A.')\B
        @test_approx_eq Ac_ldiv_B!(A, B, X) full(A')\B
    end

end
