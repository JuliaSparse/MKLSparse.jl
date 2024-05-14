using MKLSparse
using Test, SparseArrays, LinearAlgebra

sA = sprand(5, 5, 0.01)
sS = sA' * sA
sTl = tril(sS)
sTu = triu(sS)

@test MKLSparse.matdescra(Symmetric(sTl,:L)) == "SLNF"
@test MKLSparse.matdescra(Symmetric(sTu,:U)) == "SUNF"
@test MKLSparse.matdescra(Hermitian(sTl,:L)) == "HLNF"
@test MKLSparse.matdescra(Hermitian(sTu,:U)) == "HUNF"
@test MKLSparse.matdescra(LowerTriangular(sTl)) == "TLNF"
@test MKLSparse.matdescra(UpperTriangular(sTu)) == "TUNF"
@test MKLSparse.matdescra(UnitLowerTriangular(sTl)) == "TLUF"
@test MKLSparse.matdescra(UnitUpperTriangular(sTu)) == "TUUF"
@test MKLSparse.matdescra(sA) == "GFNF"

#=macro test(ex)
    return quote
        MKLSparse.__counter[] = 0
        @test $(esc(ex))
        @test MKLSparse.__counter[] == 1
    end
end=#

@testset "matrix-vector multiplication (non-square)" begin
    T = Float64
    @testset "interface = $interface" for interface in ("LP64", "ILP64")
        S = interface == "LP64" ? Int32 : Int64
        for i = 1:5
            a = sprand(T, 10, 5, 0.5)
            a = SparseMatrixCSC{T, S}(a)
            b = rand(T, 5)
            @test maximum(abs.(a*b - Array(a)*b)) < 100*eps(T)
            b = rand(T, 5, 5)
            @test maximum(abs.(a*b - Array(a)*b)) < 100*eps(T)
            b = rand(T, 10)
            @test maximum(abs.(a'*b - Array(a)'*b)) < 100*eps(T)
            @test maximum(abs.(transpose(a)*b - Array(a)'*b)) < 100*eps(T)
            b = rand(T, 10, 10)
            @test maximum(abs.(a'*b - Array(a)'*b)) < 100*eps()
            @test maximum(abs.(transpose(a)*b - Array(a)'*b)) < 100*eps(T)
        end
    end
end

#?
@testset "complex matrix-vector multiplication" begin
    T = ComplexF64
    R = Float64
    @testset "interface = $interface" for interface in ("LP64", "ILP64")
        S = interface == "LP64" ? Int32 : Int64
        for i = 1:5
            a = I + im * 0.1 * sprandn(T, 5, 5, 0.2)
            a = SparseMatrixCSC{T,S}(a)
            b = randn(T, 5, 3) + im * randn(T, 5, 3)
            c = randn(T, 5) + im * randn(T, 5)
            d = randn(T, 5) + im * randn(T, 5)
            α = rand(T)
            β = rand(T)
            @test (maximum(abs.(a*b - Array(a)*b)) < 100*eps(R))
            @test (maximum(abs.(a'*b - Array(a)'*b)) < 100*eps(R))
            @test (maximum(abs.(transpose(a)*b - transpose(Array(a))*b)) < 100*eps(R))
            @test (maximum(abs.(mul!(similar(b), a, b) - Array(a)*b)) < 100*eps(R))
            @test (maximum(abs.(mul!(similar(c), a, c) - Array(a)*c)) < 100*eps(R))
            @test (maximum(abs.(mul!(similar(b), transpose(a), b) - transpose(Array(a))*b)) < 100*eps(R))
            @test (maximum(abs.(mul!(similar(c), transpose(a), c) - transpose(Array(a))*c)) < 100*eps(R))
            @test (maximum(abs.(mul!(copy(b), a, b, α, β) - (α*(Array(a)*b) + β*b))) < 100*eps(R))
            @test (maximum(abs.(mul!(copy(b), transpose(a), b, α, β) - (α*(transpose(Array(a))*b) + β*b))) < 100*eps(R))
            @test (maximum(abs.(mul!(copy(c), transpose(a), c, α, β) - (α*(transpose(Array(a))*c) + β*c))) < 100*eps(R))
            α = β = 1 # test conversion to float
            @test (maximum(abs.(mul!(copy(b), a, b, α, β) - (α*(Array(a)*b) + β*b))) < 100*eps(R))
            @test (maximum(abs.(mul!(copy(b), transpose(a), b, α, β) - (α*(transpose(Array(a))*b) + β*b))) < 100*eps(R))
            @test (maximum(abs.(mul!(copy(c), transpose(a), c, α, β) - (α*(transpose(Array(a))*c) + β*c))) < 100*eps(R))

            c = randn(T, 6) + im * randn(T, 6)
            @test_throws DimensionMismatch transpose(a)*c
            @test_throws DimensionMismatch a.*c
            @test_throws DimensionMismatch a.*c
        end
    end
end

@testset "triangular" begin
    @testset "T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "interface = $interface" for interface in ("LP64", "ILP64")
            S = interface == "LP64" ? Int32 : Int64
            n = 100
            A = sprandn(T, n, n, 0.5) + sqrt(n)*I
            A = SparseMatrixCSC{T,S}(A)
            b = rand(T, n)
            trilA = tril(A)
            triuA = triu(A)
            trilUA = tril(A, -1) + I
            triuUA = triu(A, 1)  + I

            @test LowerTriangular(trilA) \ b ≈ Array(LowerTriangular(trilA)) \ b
            @test LowerTriangular(trilA) * b ≈ Array(LowerTriangular(trilA)) * b

            @test UpperTriangular(triuA) \ b ≈ Array(UpperTriangular(triuA)) \ b
            @test UpperTriangular(triuA) * b ≈ Array(UpperTriangular(triuA)) * b

            @test UnitLowerTriangular(trilUA) \ b ≈ Array(UnitLowerTriangular(trilUA)) \ b
            @test UnitLowerTriangular(trilUA) * b ≈ Array(UnitLowerTriangular(trilUA)) * b

            @test UnitUpperTriangular(triuUA) \ b ≈ Array(UnitUpperTriangular(triuUA)) \ b
            @test UnitUpperTriangular(triuUA) * b ≈ Array(UnitUpperTriangular(triuUA)) * b
        end
    end
end

@testset "Symmetric -- Hermitian" begin
    @testset "T = $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "interface = $interface" for interface in ("LP64", "ILP64")
            S = interface == "LP64" ? Int32 : Int64
            n = 100
            A = sprandn(T, n, n, 0.5) + sqrt(n)*I
            b = rand(T, n)
            symA = A + transpose(A)
            hermA = A + adjoint(A)

            @test Symmetric(symA) * b ≈ Array(Symmetric(symA)) * b
            @test Hermitian(hermA) * b ≈ Array(Hermitian(hermA)) * b
        end
    end
end
