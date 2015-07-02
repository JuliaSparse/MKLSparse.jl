srand(1234)

import Base.LinAlg: factorize,
       A_ldiv_B!, At_ldiv_B!, Ac_ldiv_B!, A_ldiv_B, At_ldiv_B, Ac_ldiv_B
import MKLSparse.DSS: MatrixSymStructure, DSSError,
                      cholfact, ldltfact, lufact

for T in (Float32, Float64, Complex64, Complex128)
    n = 5
    A1 = sparse(rand(T, n, n))
    A2 = A1 + A1'
    A3 = A1 + A1.'
    A4 = A1'A1
    B = rand(T, n, n)
    X = similar(B)

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

        @test_approx_eq A_ldiv_B(A, B) full(A)\B
        @test_approx_eq At_ldiv_B(A, B) full(A.')\B
        @test_approx_eq Ac_ldiv_B(A, B) full(A')\B

        for fact in (factorize, lufact, ldltfact, cholfact)
            # If factorization succeeds it should give correct answer.
            fact_failed = false
            F = 0.0 # To put F in scope... maybe better way to do this?
            try
                F = fact(A)
            catch e
                if isa(e, ArgumentError) || isa(e, DSSError)
                    fact_failed = true
                else
                    rethrow(e)
                end
            end

            if fact_failed
                continue
            end

            @test_approx_eq A_ldiv_B!(F, B, X) full(A)\B
            @test_approx_eq At_ldiv_B!(F, B, X) full(A.')\B
            @test_approx_eq Ac_ldiv_B!(F, B, X) full(A')\B

            @test_approx_eq A_ldiv_B(F, B) full(A)\B
            @test_approx_eq At_ldiv_B(F, B) full(A.')\B
            @test_approx_eq Ac_ldiv_B(F, B) full(A')\B
        end
    end

    @test_throws DimensionMismatch A_ldiv_B!(A1, rand(T, n, n+1), X)
    @test_throws DimensionMismatch A_ldiv_B!(A1, B, rand(T, n, n+1))
    @test_throws DimensionMismatch A_ldiv_B!(sparse(rand(T, n+1, n+1)), B, X)

    @test_throws DimensionMismatch A_ldiv_B(A1, rand(T, n+1, n))
    @test_throws DimensionMismatch A_ldiv_B(sparse(rand(T, n+1, n+1)), B)
    @test_throws DimensionMismatch A_ldiv_B(sparse(rand(T, n, n+1)), B)

end

for T in (Float32, Float64, Complex64, Complex128)
    n = 5
    A1 = sparse(rand(T, n, n))
    A2 = A1 + A1'
    A3 = A1 + A1.'
    A4 = A1'A1
    B = rand(T, n, n)
    X = similar(B)

    @test_throws ArgumentError cholfact(A1)

    if T <: Complex
        # Will have complex diagonal so not candidate for chol
        #
        #@test_throws DSSError cholfact(A2)
        @test_throws ArgumentError cholfact(A3)
    else
        # Will not be pos def
        @test_throws DSSError cholfact(A2)
        @test_throws DSSError cholfact(A2)
    end

    if T <: Complex
        @test_throws ArgumentError ldltfact(A1)
        @test_throws ArgumentError ldltfact(A3)
    end
end
