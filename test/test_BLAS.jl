module Matdescra

using MKLSparse
using Base.Test

sA = sprand(5, 5, 0.01)
sS = sA'sA
sTl = tril(sS)
sTu = triu(sS)

@test MKLSparse.BLAS.matdescra(Base.LinAlg.Symmetric(sTl, :L)) == "SLNF"
@test MKLSparse.BLAS.matdescra(Base.LinAlg.Symmetric(sTu, :U)) == "SUNF"
@test MKLSparse.BLAS.matdescra(Base.LinAlg.LowerTriangular(sTl)) == "TLNF"
@test MKLSparse.BLAS.matdescra(Base.LinAlg.UpperTriangular(sTu)) == "TUNF"
@test MKLSparse.BLAS.matdescra(Base.LinAlg.UnitLowerTriangular(sTl)) == "TLUF"
@test MKLSparse.BLAS.matdescra(Base.LinAlg.UnitUpperTriangular(sTu)) == "TUUF"
@test MKLSparse.BLAS.matdescra(sA) == "GUUF"

end  # module

module BLAS

import Base.LinAlg: UnitLowerTriangular, UnitUpperTriangular


using MKLSparse
using Base.Test

macro test_blas(ex)
    return quote
        MKLSparse.BLAS.__counter[] = 0
        @test $(esc(ex))
        @test MKLSparse.BLAS.__counter[] == 1
    end
end

@testset "matrix-vector multiplication (non-square)" begin
    for i = 1:5
        a = sprand(10, 5, 0.5)
        b = rand(5)
        @test_blas maximum(abs.(a * b - Array(a) * b)) < 100 * eps()
        b = rand(5, 5)
        @test_blas maximum(abs.(a * b - Array(a) * b)) < 100 * eps()
        b = rand(10)
        @test_blas maximum(abs.(a' * b - Array(a)' * b)) < 100 * eps()
        @test_blas maximum(abs.(a.' * b - Array(a)' * b)) < 100 * eps()
        b = rand(10, 10)
        @test_blas maximum(abs.(a' * b - Array(a)' * b)) < 100 * eps()
        @test_blas maximum(abs.(a.' * b - Array(a)' * b)) < 100 * eps()
    end
end

#?
@testset "complex matrix-vector multiplication" begin
    for i = 1:5
        a = speye(5) + im * 0.1 * sprandn(5, 5, 0.2)
        b = randn(5, 3) + im * randn(5, 3)
        c = randn(5) + im * randn(5)
        d = randn(5) + im * randn(5)
        α = rand(Complex128)
        β = rand(Complex128)
        @test_blas (maximum(abs.(a * b - Array(a) * b)) < 100 * eps())
        @test_blas (maximum(abs.(a' * b - Array(a)' * b)) < 100 * eps())
        @test_blas (maximum(abs.(a.' * b - Array(a).' * b)) < 100 * eps())
        @test_blas (maximum(abs.(A_mul_B!(similar(b), a, b) - Array(a) * b)) < 100 * eps())
        @test_blas (maximum(abs.(A_mul_B!(similar(c), a, c) - Array(a) * c)) < 100 * eps())
        @test_blas (maximum(abs.(At_mul_B!(similar(b), a, b) - Array(a).' * b)) < 100 * eps())
        @test_blas (maximum(abs.(At_mul_B!(similar(c), a, c) - Array(a).' * c)) < 100 * eps())

        c = randn(6) + im * randn(6)
        @test_throws DimensionMismatch a.' * c
        @test_throws DimensionMismatch a .* c
        @test_throws DimensionMismatch a .* c
    end
end

@testset "triangular" begin
    n = 100
    A = sprandn(n, n, 0.5) + sqrt(n) * I
    b = rand(n)
    symA = A + A.'
    trilA = tril(A)
    triuA = triu(A)
    trilUA = tril(A, -1) + speye(A)
    triuUA = triu(A, 1) + speye(A)

    @test_blas LowerTriangular(trilA) \ b ≈ Array(LowerTriangular(trilA)) \ b
    @test_blas LowerTriangular(trilA) * b ≈ Array(LowerTriangular(trilA)) * b

    @test_blas UpperTriangular(triuA) \ b ≈ Array(UpperTriangular(triuA)) \ b
    @test_blas UpperTriangular(triuA) * b ≈ Array(UpperTriangular(triuA)) * b

    @test_blas UnitLowerTriangular(trilUA) \ b ≈ Array(UnitLowerTriangular(trilUA)) \ b
    @test_blas UnitLowerTriangular(trilUA) * b ≈ Array(UnitLowerTriangular(trilUA)) * b

    @test_blas UnitUpperTriangular(triuUA) \ b ≈ Array(UnitUpperTriangular(triuUA)) \ b
    @test_blas UnitUpperTriangular(triuUA) * b ≈ Array(UnitUpperTriangular(triuUA)) * b

    @test_blas Symmetric(symA) * b ≈ Array(Symmetric(symA)) * b
end

end # module
