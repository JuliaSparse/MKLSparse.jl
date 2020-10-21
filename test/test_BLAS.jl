using MKLSparse
using Test, SparseArrays, LinearAlgebra

macro test_blas(ex)
    return quote
        MKLSparse.BLAS.__counter[] = 0
        @test $(esc(ex))
        @test MKLSparse.BLAS.__counter[] == 1
    end
end

@testset "matrix-vector muutiplication (non-square)" begin
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
@testset "complex matrix-vector muutiplication" begin
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
    bmat = reshape(b, (:, 1)) # Matrix with 1 column
    symA = A + transpose(A)
    trilA = tril(A)
    triuA = triu(A)
    trilUA = tril(A, -1) + I
    triuUA = triu(A, 1)  + I

    ltA = LowerTriangular(trilA)
    trilAd = Array(trilA)          # trilA as a dense matrix
    ltAd = LowerTriangular(trilAd) # ltA as a LowerTriangular dense matrix
    @test_blas ltA \ b ≈ ltAd \ b
    @test_blas ltA \ bmat ≈ ltAd \ bmat
    @test_blas ltA' \ b ≈ ltAd' \ b
    @test_blas ltA' \ bmat ≈ ltAd' \ bmat
    @test_blas ltA * b ≈ ltAd * b
    @test_blas ltA * bmat ≈ ltAd * bmat
    @test_blas trilA * b ≈ trilAd * b
    @test_blas trilA * bmat ≈ trilAd * bmat
    @test_blas ltA' * b ≈ ltAd' * b
    @test_blas ltA' * bmat ≈ ltAd' * bmat
    @test_blas trilA' * b ≈ trilAd' * b
    @test_blas trilA' * bmat ≈ trilAd' * bmat

    utA = UpperTriangular(triuA)
    triuAd = Array(triuA)          # triuA as a dense matrix
    utAd = UpperTriangular(triuAd) # utA as a UpperTriangular dense matrix
    @test_blas utA \ b ≈ utAd \ b
    @test_blas utA \ bmat ≈ utAd \ bmat
    @test_blas utA' \ b ≈ utAd' \ b
    @test_blas utA' \ bmat ≈ utAd' \ bmat
    @test_blas utA * b ≈ utAd * b
    @test_blas utA * bmat ≈ utAd * bmat
    @test_blas trilA * b ≈ trilAd * b
    @test_blas trilA * bmat ≈ trilAd * bmat
    @test_blas utA' * b ≈ utAd' * b
    @test_blas utA' * bmat ≈ utAd' * bmat
    @test_blas triuA' * b ≈ triuAd' * b
    @test_blas triuA' * bmat ≈ triuAd' * bmat

    ltUA = UnitLowerTriangular(trilUA)
    trilUAd = Array(trilUA)        # trilUA as a dense matrix
    ltUAd = UnitLowerTriangular(trilUAd)
    @test_blas ltUA \ b ≈ ltUAd \ b
    @test_blas ltUA \ bmat ≈ ltUAd \ bmat
    @test_blas ltUA' \ b ≈ ltUAd' \ b
    @test_blas ltUA' \ bmat ≈ ltUAd' \ bmat
    @test_blas ltUA * b ≈ ltUAd * b
    @test_blas ltUA * bmat ≈ ltUAd * bmat
    @test_blas trilUA * b ≈ ltUAd * b
    @test_blas trilUA * bmat ≈ ltUAd * bmat
    @test_blas ltUA' * b ≈ ltUAd' * b
    @test_blas ltUA' * bmat ≈ ltUAd' * bmat
    @test_blas trilUA' * b ≈ trilUAd' * b
    @test_blas trilUA' * bmat ≈ trilUAd' * bmat

    utUA = UnitUpperTriangular(triuUA)
    triuUAd = Array(triuUA)        # triuUA as a dense matrix
    utUAd = UnitUpperTriangular(triuUAd)
    @test_blas utUA \ b ≈ utUAd \ b
    @test_blas utUA \ bmat ≈ utUAd \ bmat
    @test_blas utUA' \ b ≈ utUAd' \ b
    @test_blas utUA' \ bmat ≈ utUAd' \ bmat
    @test_blas utUA * b ≈ utUAd * b
    @test_blas utUA * bmat ≈ utUAd * bmat
    @test_blas triuUA * b ≈ utUAd * b
    @test_blas triuUA * bmat ≈ utUAd * bmat
    @test_blas utUA' * b ≈ utUAd' * b
    @test_blas utUA' * bmat ≈ utUAd' * bmat
    @test_blas triuUA' * b ≈ triuUAd' * b
    @test_blas triuUA' * bmat ≈ triuUAd' * bmat

    @test_blas Symmetric(symA) * b ≈ Symmetric(Array(symA)) * b
end
