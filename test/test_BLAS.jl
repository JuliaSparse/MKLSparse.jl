using MKLSparse
using Test, SparseArrays, LinearAlgebra

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

@testset "SparseBLAS for $T matrices and vectors and $IT indices" for
    T in (Float32, Float64, ComplexF32, ComplexF64),
    IT in (Base.USE_BLAS64 ? (Int32, Int64) : (Int32,))

local atol::real(T) = 100*eps(real(one(T))) # absolute tolerance for SparseBLAS results

@testset "SparseMatrixCSC{$T,$IT} * Vector{$T}" begin
    for _ in 1:10
        spA = convert(SpraseMatrixCSC{T, IT}, sprand(T, 10, 5, 0.5))
        a = Array(spA)
        b = rand(T, 5)
        c = rand(T, 10)

        @test @blas(spA*b) ≈ a*b atol=atol
        @test @blas(spA'*c) ≈ a'*c atol=atol
        @test @blas(transpose(spA)*c) ≈ transpose(a)*c atol=atol

        @test_throws DimensionMismatch spA*c
        @test_throws DimensionMismatch spA'*b
        @test_throws DimensionMismatch transpose(spA)*b
    end
end

@testset "Vector{$T} * SparseMatrixCSC{$T,$IT}" begin
    for _ in 1:10
        spA = convert(SparseMatrixCSC{T, IT}, sprand(T, 10, 5, 0.5))
        a = Array(spA)
        b = rand(T, 10)
        c = rand(T, 5)

        @test @blas(b'*spA) ≈ b'*a atol=atol
        @test @blas(c'*spA') ≈ c'*a' atol=atol

        @test_throws DimensionMismatch c*spA
        @test_throws DimensionMismatch b*spA'
        @test_throws DimensionMismatch b*transpose(spA)

        @test_throws DimensionMismatch c'*spA
        @test_throws DimensionMismatch b'*spA'

        if !(T <: Complex) # adjoint*transposed isn't routed to BLAS call
            @test @blas(c'*transpose(spA)) ≈ c'*transpose(a) atol=atol
            @test_throws DimensionMismatch b'*transpose(spA)
        end
    end
end

@testset "SparseMatrixCSC{$T,$IT} * Matrix{$T}" begin
    for _ in 1:10
        spA = convert(SparseMatrixCSC{T,IT}, sprand(T, 10, 5, 0.5))
        a = Array(spA)
        b = rand(T, 5, 8)
        c = rand(T, 10, 12)
        ab = rand(T, 10, 8)
        tac = rand(T, 5, 12)
        α = rand(T)
        β = rand(T)

        @test @blas(spA*b) ≈ a*b atol=atol
        @test @blas(spA'*c) ≈ a'*c atol=atol
        @test @blas(transpose(spA)*c) ≈ transpose(a)*c atol=atol

        @test_throws DimensionMismatch spA*c
        @test_throws DimensionMismatch spA'*b
        @test_throws DimensionMismatch transpose(spA)*b

        @test @blas(mul!(similar(ab), spA, b)) ≈ a*b atol=atol
        @test @blas(mul!(similar(tac), spA', c)) ≈ a'*c atol=atol
        @test @blas(mul!(similar(tac), transpose(spA), c)) ≈ transpose(a)*c atol=atol

        @test @blas(mul!(copy(ab), spA, b, α, β)) ≈ α*a*b + β*ab atol=atol
        @test @blas(mul!(copy(tac), spA', c, α, β)) ≈ α*a'*c + β*tac atol=atol
        @test @blas(mul!(copy(tac), transpose(spA), c, α, β)) ≈ α*transpose(a)*c + β*tac atol=atol

        @test @blas(mul!(copy(ab), spA, b, 1, 1)) ≈ a*b + ab atol=atol
        @test @blas(mul!(copy(tac), transpose(spA), c, 1, 1)) ≈ transpose(a)*c + tac atol=atol
        @test @blas(mul!(copy(tac), spA', c, 1, 1)) ≈ a'*c + tac atol=atol

        symA = spA + transpose(spA)
        @test @blas(Symmetric(symA) * b) ≈ Array(Symmetric(symA)) * b
        hermA = spA + adjoint(spA)
        @test @blas(Hermitian(hermA) * b) ≈ Array(Hermitian(hermA)) * b
    end
end

@testset "SparseMatrixCSC{$T, $IT} {* /} Vector{$T} for triangular/symmetric/hermitian" begin
    for _ in 1:10
        n = rand(50:150)
        spA = convert(SparseMatrixCSC{T, IT}, sprand(T, n, n, 0.5) + convert(real(T), sqrt(n))*I)
        A = Array(spA)
        b = rand(T, n)

        trilA = tril(spA)
        @test @blas(LowerTriangular(trilA) \ b) ≈ Array(LowerTriangular(trilA)) \ b
        @test @blas(LowerTriangular(trilA) * b) ≈ Array(LowerTriangular(trilA)) * b

        triuA = triu(spA)
        @test @blas(UpperTriangular(triuA) \ b) ≈ Array(UpperTriangular(triuA)) \ b
        @test @blas(UpperTriangular(triuA) * b) ≈ Array(UpperTriangular(triuA)) * b

        trilUA = tril(spA, -1) + I
        @test @blas(UnitLowerTriangular(trilUA) \ b) ≈ Array(UnitLowerTriangular(trilUA)) \ b
        @test @blas(UnitLowerTriangular(trilUA) * b) ≈ Array(UnitLowerTriangular(trilUA)) * b

        triuUA = triu(spA, 1)  + I
        @test @blas(UnitUpperTriangular(triuUA) \ b) ≈ Array(UnitUpperTriangular(triuUA)) \ b
        @test @blas(UnitUpperTriangular(triuUA) * b) ≈ Array(UnitUpperTriangular(triuUA)) * b

        symA = spA + transpose(spA)
        @test @blas(Symmetric(symA) * b) ≈ Array(Symmetric(symA)) * b

        hermA = spA + adjoint(spA)
        @test @blas(Hermitian(hermA) * b) ≈ Array(Hermitian(hermA)) * b
    end
end

end

