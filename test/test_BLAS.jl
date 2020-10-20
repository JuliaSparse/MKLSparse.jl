using MKLSparse
using Test, SparseArrays, LinearAlgebra

@testset "matdescra" begin

    sA = sprand(5, 5, 0.01)
    sS = sA'sA
    sTl = tril(sS)
    sTu = triu(sS)

    @test MKLSparse.BLAS.matdescra(Symmetric(sTl,:L)) == "SLNF"
    @test MKLSparse.BLAS.matdescra(Symmetric(sTu,:U)) == "SUNF"
    @test MKLSparse.BLAS.matdescra(LowerTriangular(sTl)) == "TLNF"
    @test MKLSparse.BLAS.matdescra(UpperTriangular(sTu)) == "TUNF"
    @test MKLSparse.BLAS.matdescra(UnitLowerTriangular(sTl)) == "TLUF"
    @test MKLSparse.BLAS.matdescra(UnitUpperTriangular(sTu)) == "TUUF"
    @test MKLSparse.BLAS.matdescra(sA) == "GUUF"

end

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
        @test_blas maximum(abs.(a*b - Array(a)*b)) < 100*eps()
        b = rand(5, 5)
        @test_blas maximum(abs.(a*b - Array(a)*b)) < 100*eps()
        b = rand(10)
        @test_blas maximum(abs.(a'*b - Array(a)'*b)) < 100*eps()
        @test_blas maximum(abs.(transpose(a)*b - Array(a)'*b)) < 100*eps()
        b = rand(10,10)
        @test_blas maximum(abs.(a'*b - Array(a)'*b)) < 100*eps()
        @test_blas maximum(abs.(transpose(a)*b - Array(a)'*b)) < 100*eps()
    end
end

#?
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

@testset "triangular" begin
    n = 100
    A = sprandn(n, n, 0.5) + sqrt(n)*I
    b = rand(n)
    symA = A + transpose(A)
    trilA = tril(A)
    triuA = triu(A)
    trilUA = tril(A, -1) + I
    triuUA = triu(A, 1)  + I

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
