using MKLSparse
using Test, SparseArrays, LinearAlgebra

# evaluates ex and checks whether it has called any SparseBLAS MKL method
macro blas(ex)
    quote
        begin
            MKLSparse.__counter[] = 0
            local res = $(esc(ex))
            @test MKLSparse.__counter[] == 1
            res
        end
    end
end

@testset "MKLSparse.matdescra()" begin
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
end

const atol = 100*eps() # absolute tolerance for SparseBLAS results

@testset "matrix-vector multiplication (non-square)" begin
    for i = 1:5
        a = sprand(10, 5, 0.5)
        b = rand(5)
        @test @blas(a*b) ≈ Array(a)*b atol=atol
        b = rand(5, 5)
        @test @blas(a*b) ≈ Array(a)*b atol=atol
        b = rand(10)
        @test @blas(a'*b) ≈ Array(a)'*b atol=atol
        @test @blas(transpose(a)*b) ≈ Array(a)'*b atol=atol
        b = rand(10,10)
        @test @blas(a'*b) ≈ Array(a)'*b atol=atol
        @test @blas(transpose(a)*b) ≈ Array(a)'*b atol=atol
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
        @test @blas(a*b) ≈ Array(a)*b atol=atol
        @test @blas(a'*b) ≈ Array(a)'*b atol=atol
        @test @blas(transpose(a)*b) ≈ transpose(Array(a))*b atol=atol
        @test @blas(mul!(similar(b), a, b)) ≈ Array(a)*b atol=atol
        @test @blas(mul!(similar(c), a, c)) ≈ Array(a)*c atol=atol
        @test @blas(mul!(similar(b), transpose(a), b)) ≈ transpose(Array(a))*b atol=atol
        @test @blas(mul!(similar(c), transpose(a), c)) ≈ transpose(Array(a))*c atol=atol
        @test @blas(mul!(copy(b), a, b, α, β)) ≈ (α*(Array(a)*b) + β*b) atol=atol
        @test @blas(mul!(copy(b), transpose(a), b, α, β)) ≈ (α*(transpose(Array(a))*b) + β*b) atol=atol
        @test @blas(mul!(copy(c), transpose(a), c, α, β)) ≈ (α*(transpose(Array(a))*c) + β*c) atol=atol
        α = β = 1 # test conversion to float
        @test @blas(mul!(copy(b), a, b, α, β)) ≈ (α*(Array(a)*b) + β*b) atol=atol
        @test @blas(mul!(copy(b), transpose(a), b, α, β)) ≈ (α*(transpose(Array(a))*b) + β*b) atol=atol
        @test @blas(mul!(copy(c), transpose(a), c, α, β)) ≈ (α*(transpose(Array(a))*c) + β*c) atol=atol

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

    @test @blas(LowerTriangular(trilA) \ b) ≈ Array(LowerTriangular(trilA)) \ b
    @test @blas(LowerTriangular(trilA) * b) ≈ Array(LowerTriangular(trilA)) * b

    @test @blas(UpperTriangular(triuA) \ b) ≈ Array(UpperTriangular(triuA)) \ b
    @test @blas(UpperTriangular(triuA) * b) ≈ Array(UpperTriangular(triuA)) * b

    @test @blas(UnitLowerTriangular(trilUA) \ b) ≈ Array(UnitLowerTriangular(trilUA)) \ b
    @test @blas(UnitLowerTriangular(trilUA) * b) ≈ Array(UnitLowerTriangular(trilUA)) * b

    @test @blas(UnitUpperTriangular(triuUA) \ b) ≈ Array(UnitUpperTriangular(triuUA)) \ b
    @test @blas(UnitUpperTriangular(triuUA) * b) ≈ Array(UnitUpperTriangular(triuUA)) * b

    @test @blas(Symmetric(symA) * b) ≈ Array(Symmetric(symA)) * b
end
