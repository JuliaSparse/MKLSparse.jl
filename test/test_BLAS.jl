using MKLSparse
using Test, SparseArrays, LinearAlgebra

@testset "MKLSparse.matdescra()" begin
    sA = sprand(5, 5, 0.01)
    sS = sA'sA
    sTl = tril(sS)
    sTu = triu(sS)

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

# evaluates ex and checks whether it has called any SparseBLAS MKL method
macro blas(ex)
    quote
        begin
            MKLSparse.__counter[] = 0
            local res = $(@inferred(esc(ex)))
            @test MKLSparse.__counter[] == 1
            res
        end
    end
end

for T in (Float64, ComplexF64)
  INT_TYPES = Base.USE_BLAS64 ? (Int32, Int64) : (Int32,)
  for INT in INT_TYPES
        @testset "matrix-vector and matrix-matrix multiplications (non-square) -- $T -- $INT" begin
            for i = 1:5
                A = sprand(T, 10, 5, 0.5)
                A = SparseMatrixCSC{T,INT}(A)
                b = rand(T, 5)
                @test @blas(A*b) ≈ Array(A)*b
                B = rand(T, 5, 5)
                @test @blas(A*B) ≈ Array(A)*B
                b = rand(T, 10)
                @test @blas(A'*b) ≈ Array(A)'*b
                @test @blas(transpose(A)*b) ≈ transpose(Array(A))*b
                B = rand(T, 10, 10)
                @test @blas(A'*B) ≈ Array(A)'*B
                @test @blas(transpose(A)*B) ≈ transpose(Array(A))*B
            end
        end

        @testset "Symmetric / Hermitian -- $T -- $INT" begin
            n = 10
            A = sprandn(T, n, n, 0.5) + sqrt(n)*I
            A = SparseMatrixCSC{T,INT}(A)
            b = rand(T, n)
            B = rand(T, n, 3)
            symA = A + transpose(A)
            hermA = A + adjoint(A)
            @test @blas(Symmetric(symA) * b) ≈ Array(Symmetric(symA)) * b
            @test @blas(Hermitian(hermA) * b) ≈ Array(Hermitian(hermA)) * b
            @test @blas(Symmetric(symA) * B) ≈ Array(Symmetric(symA)) * B
            @test @blas(Hermitian(hermA) * B) ≈ Array(Hermitian(hermA)) * B
        end

        @testset "triangular -- $T -- $INT" begin
            n = 10
            A = sprandn(T, n, n, 0.5) + sqrt(n)*I
            A = SparseMatrixCSC{T,INT}(A)
            b = rand(T, n)
            B = rand(T, n, 3)
            trilA = tril(A)
            triuA = triu(A)
            trilUA = tril(A, -1) + I
            triuUA = triu(A, 1)  + I

            @test @blas(LowerTriangular(trilA) \ b) ≈ Array(LowerTriangular(trilA)) \ b
            @test @blas(LowerTriangular(trilA) * b) ≈ Array(LowerTriangular(trilA)) * b
            @test @blas(LowerTriangular(trilA) \ B) ≈ Array(LowerTriangular(trilA)) \ B
            @test @blas(LowerTriangular(trilA) * B) ≈ Array(LowerTriangular(trilA)) * B

            @test @blas(UpperTriangular(triuA) \ b) ≈ Array(UpperTriangular(triuA)) \ b
            @test @blas(UpperTriangular(triuA) * b) ≈ Array(UpperTriangular(triuA)) * b
            @test @blas(UpperTriangular(triuA) \ B) ≈ Array(UpperTriangular(triuA)) \ B
            @test @blas(UpperTriangular(triuA) * B) ≈ Array(UpperTriangular(triuA)) * B

            @test @blas(UnitLowerTriangular(trilUA) \ b) ≈ Array(UnitLowerTriangular(trilUA)) \ b
            @test @blas(UnitLowerTriangular(trilUA) * b) ≈ Array(UnitLowerTriangular(trilUA)) * b
            @test @blas(UnitLowerTriangular(trilUA) \ B) ≈ Array(UnitLowerTriangular(trilUA)) \ B
            @test @blas(UnitLowerTriangular(trilUA) * B) ≈ Array(UnitLowerTriangular(trilUA)) * B

            @test @blas(UnitUpperTriangular(triuUA) \ b) ≈ Array(UnitUpperTriangular(triuUA)) \ b
            @test @blas(UnitUpperTriangular(triuUA) * b) ≈ Array(UnitUpperTriangular(triuUA)) * b
            @test @blas(UnitUpperTriangular(triuUA) \ B) ≈ Array(UnitUpperTriangular(triuUA)) \ B
            @test @blas(UnitUpperTriangular(triuUA) * B) ≈ Array(UnitUpperTriangular(triuUA)) * B
        end
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
        @test (maximum(abs.(@blas(a*b) - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(@blas(a'*b) - Array(a)'*b)) < 100*eps())
        @test (maximum(abs.(@blas(transpose(a)*b) - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(@blas(mul!(similar(b), a, b)) - Array(a)*b)) < 100*eps())
        @test (maximum(abs.(@blas(mul!(similar(c), a, c)) - Array(a)*c)) < 100*eps())
        @test (maximum(abs.(@blas(mul!(similar(b), transpose(a), b)) - transpose(Array(a))*b)) < 100*eps())
        @test (maximum(abs.(@blas(mul!(similar(c), transpose(a), c)) - transpose(Array(a))*c)) < 100*eps())
        @test (maximum(abs.(@blas(mul!(copy(b), a, b, α, β)) - (α*(Array(a)*b) + β*b))) < 100*eps())
        @test (maximum(abs.(@blas(mul!(copy(b), transpose(a), b, α, β)) - (α*(transpose(Array(a))*b) + β*b))) < 100*eps())
        @test (maximum(abs.(@blas(mul!(copy(c), transpose(a), c, α, β)) - (α*(transpose(Array(a))*c) + β*c))) < 100*eps())
        α = β = 1 # test conversion to float
        @test (maximum(abs.(@blas(mul!(copy(b), a, b, α, β)) - (α*(Array(a)*b) + β*b))) < 100*eps())
        @test (maximum(abs.(@blas(mul!(copy(b), transpose(a), b, α, β)) - (α*(transpose(Array(a))*b) + β*b))) < 100*eps())
        @test (maximum(abs.(@blas(mul!(copy(c), transpose(a), c, α, β)) - (α*(transpose(Array(a))*c) + β*c))) < 100*eps())

        c = randn(6) + im*randn(6)
        @test_throws DimensionMismatch transpose(a)*c
        @test_throws DimensionMismatch a.*c
        @test_throws DimensionMismatch a.*c
    end
end
