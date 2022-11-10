using MKLSparse
using Test, SparseArrays, LinearAlgebra

sA = sprand(5, 5, 0.01)
sS = sA'sA
sTl = tril(sS)
sTu = triu(sS)

@test MKLSparse.matdescra(Symmetric(sTl,:L)) == "SLNF"
@test MKLSparse.matdescra(Symmetric(sTu,:U)) == "SUNF"
@test MKLSparse.matdescra(LowerTriangular(sTl)) == "TLNF"
@test MKLSparse.matdescra(UpperTriangular(sTu)) == "TUNF"
@test MKLSparse.matdescra(UnitLowerTriangular(sTl)) == "TLUF"
@test MKLSparse.matdescra(UnitUpperTriangular(sTu)) == "TUUF"
@test MKLSparse.matdescra(sA) == "GUUF"

macro test_blas(ex)
    return quote
        MKLSparse.__counter[] = 0
        @test $(esc(ex))
        @test MKLSparse.__counter[] == 1
    end
end

for T in (Float64, ComplexF64)
    @testset "matrix-vector and matrix-matrix multiplications (non-square) -- $T" begin
        for i = 1:5
            a = sprand(T, 10, 5, 0.5)
            b = rand(T, 5)
            @test_blas a*b ≈ Array(a)*b
            B = rand(T, 5, 5)
            @test_blas a*B ≈ Array(a)*B
            b = rand(T, 10)
            @test_blas a'*b ≈ Array(a)'*b
            @test_blas transpose(a)*b ≈ transpose(Array(a))*b
            B = rand(T, 10, 10)
            @test_blas a'*B ≈ Array(a)'*B
            @test_blas transpose(a)*B ≈ transpose(Array(a))*B
        end
    end

    @testset "Symmetric / Hermitian -- $T" begin
        n = 10
        A = sprandn(T, n, n, 0.5) + sqrt(n)*I
        b = rand(T, n)
        B = rand(T, n, 3)
        symA = A + transpose(A)
        hermA = A + adjoint(A)
        @test_blas Symmetric(symA) * b ≈ Array(Symmetric(symA)) * b
        @test_blas Hermitian(hermA) * b ≈ Array(Hermitian(hermA)) * b
        @test_blas Symmetric(symA) * B ≈ Array(Symmetric(symA)) * B
        @test_blas Hermitian(hermA) * B ≈ Array(Hermitian(hermA)) * B
    end

    @testset "triangular -- $T" begin
        n = 10
        A = sprandn(T, n, n, 0.5) + sqrt(n)*I
        b = rand(T, n)
        B = rand(T, n, 3)
        trilA = tril(A)
        triuA = triu(A)
        trilUA = tril(A, -1) + I
        triuUA = triu(A, 1)  + I

        @test_blas LowerTriangular(trilA) \ b ≈ Array(LowerTriangular(trilA)) \ b
        @test_blas LowerTriangular(trilA) * b ≈ Array(LowerTriangular(trilA)) * b
        @test_blas LowerTriangular(trilA) \ B ≈ Array(LowerTriangular(trilA)) \ B
        @test_blas LowerTriangular(trilA) * B ≈ Array(LowerTriangular(trilA)) * B

        @test_blas UpperTriangular(triuA) \ b ≈ Array(UpperTriangular(triuA)) \ b
        @test_blas UpperTriangular(triuA) * b ≈ Array(UpperTriangular(triuA)) * b
        @test_blas UpperTriangular(triuA) \ B ≈ Array(UpperTriangular(triuA)) \ B
        @test_blas UpperTriangular(triuA) * B ≈ Array(UpperTriangular(triuA)) * B

        @test_blas UnitLowerTriangular(trilUA) \ b ≈ Array(UnitLowerTriangular(trilUA)) \ b
        @test_blas UnitLowerTriangular(trilUA) * b ≈ Array(UnitLowerTriangular(trilUA)) * b
        @test_blas UnitLowerTriangular(trilUA) \ B ≈ Array(UnitLowerTriangular(trilUA)) \ B
        @test_blas UnitLowerTriangular(trilUA) * B ≈ Array(UnitLowerTriangular(trilUA)) * B

        @test_blas UnitUpperTriangular(triuUA) \ b ≈ Array(UnitUpperTriangular(triuUA)) \ b
        @test_blas UnitUpperTriangular(triuUA) * b ≈ Array(UnitUpperTriangular(triuUA)) * b
        @test_blas UnitUpperTriangular(triuUA) \ B ≈ Array(UnitUpperTriangular(triuUA)) \ B
        @test_blas UnitUpperTriangular(triuUA) * B ≈ Array(UnitUpperTriangular(triuUA)) * B
    end
end

@testset "complex matrix-vector multiplication" begin
    for i = 1:5
        a = I + im * 0.1*sprandn(5, 5, 0.2)
        b = randn(5,3) + im*randn(5,3)
        c = randn(5) + im*randn(5)
        d = randn(5) + im*randn(5)
        α = rand(ComplexF64)
        β = rand(ComplexF64)
        @test_blas (maximum(abs.(a*b - Array(a)*b)) < 100*eps())
        @test_blas (maximum(abs.(a'*b - Array(a)'*b)) < 100*eps())
        @test_blas (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps())
        @test_blas (maximum(abs.(mul!(similar(b), a, b) - Array(a)*b)) < 100*eps())
        @test_blas (maximum(abs.(mul!(similar(c), a, c) - Array(a)*c)) < 100*eps())
        @test_blas (maximum(abs.(mul!(similar(b), transpose(a), b) - transpose(Array(a))*b)) < 100*eps())
        @test_blas (maximum(abs.(mul!(similar(c), transpose(a), c) - transpose(Array(a))*c)) < 100*eps())
        @test_blas (maximum(abs.(mul!(copy(b), a, b, α, β) - (α*(Array(a)*b) + β*b))) < 100*eps())
        @test_blas (maximum(abs.(mul!(copy(b), transpose(a), b, α, β) - (α*(transpose(Array(a))*b) + β*b))) < 100*eps())
        @test_blas (maximum(abs.(mul!(copy(c), transpose(a), c, α, β) - (α*(transpose(Array(a))*c) + β*c))) < 100*eps())
        α = β = 1 # test conversion to float
        @test_blas (maximum(abs.(mul!(copy(b), a, b, α, β) - (α*(Array(a)*b) + β*b))) < 100*eps())
        @test_blas (maximum(abs.(mul!(copy(b), transpose(a), b, α, β) - (α*(transpose(Array(a))*b) + β*b))) < 100*eps())
        @test_blas (maximum(abs.(mul!(copy(c), transpose(a), c, α, β) - (α*(transpose(Array(a))*c) + β*c))) < 100*eps())

        c = randn(6) + im*randn(6)
        @test_throws DimensionMismatch transpose(a)*c
        @test_throws DimensionMismatch a.*c
        @test_throws DimensionMismatch a.*c
    end
end
