using MKLSparse
using Test, SparseArrays, LinearAlgebra

# evaluates ex and checks whether it has called any SparseBLAS MKL method
macro blas(ex)
    quote
        begin
            MKLSparse.__mklsparse_calls_count[] = 0
            local res = $(esc(ex))
            @test MKLSparse.__mklsparse_calls_count[] == 1
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

@testset "SparseBLAS for $T matrices and vectors" for T in (Float32, Float64, ComplexF32, ComplexF64)

local atol::real(T) = 100*eps(real(one(T))) # absolute tolerance for SparseBLAS results

@testset "SparseMatrixCSC{$T} * Vector{$T}" begin
    for _ in 1:10
        spA = sprand(T, 10, 5, 0.5)
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

@testset "Vector{$T} * SparseMatrixCSC{$T}" begin
    for _ in 1:10
        spA = sprand(T, 10, 5, 0.5)
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

@testset "SparseMatrixCSC{$T} * Matrix{$T}" begin
    for _ in 1:10
        spA = sprand(T, 10, 5, 0.5)
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
    end
end

@testset "Matrix{$T} * SparseMatrixCSC{$T}" begin
    for _ in 1:10
        spA = sprand(T, 10, 5, 0.5)
        a = Array(spA)
        b = rand(T, 12, 10)
        c = rand(T, 8, 5)
        ba = rand(T, 12, 5)
        cta = rand(T, 8, 10)
        α = rand(T)
        β = rand(T)

        # on MacOS nightly Julia b*spA is broken (only the first column is correct)
        broken_bXspA = Sys.isapple() && (VERSION > v"1.8")

        # the VERSION branching is a workaround for @test not supporting broken= before 1.7
        # currently there are no broken pre-1.7 tests, so we just remove broken= for these versions
        if VERSION >= v"1.7"
            @test @blas(b*spA) ≈ b*a atol=atol broken=broken_bXspA
        else
            @test @blas(b*spA) ≈ b*a atol=atol
        end
        @test @blas(c*spA') ≈ c*a' atol=atol
        @test @blas(c*transpose(spA)) ≈ c*transpose(a) atol=atol

        @test_throws DimensionMismatch c*spA
        @test_throws DimensionMismatch b*spA'
        @test_throws DimensionMismatch b*transpose(spA)

        if VERSION >= v"1.7"
            @test @blas(mul!(similar(ba), b, spA)) ≈ b*a atol=atol broken=broken_bXspA
        else
            @test @blas(mul!(similar(ba), b, spA)) ≈ b*a atol=atol
        end
        @test @blas(mul!(similar(cta), c, spA')) ≈ c*a' atol=atol
        @test @blas(mul!(similar(cta), c, transpose(spA))) ≈ c*transpose(a) atol=atol

        if VERSION >= v"1.7"
            @test @blas(mul!(copy(ba), b, spA, α, β)) ≈ α*b*a + β*ba atol=atol broken=broken_bXspA
        else
            @test @blas(mul!(copy(ba), b, spA, α, β)) ≈ α*b*a + β*ba atol=atol
        end
        @test @blas(mul!(copy(cta), c, spA', α, β)) ≈ α*c*a' + β*cta atol=atol
        @test @blas(mul!(copy(cta), c, transpose(spA), α, β)) ≈ α*c*transpose(a) + β*cta atol=atol

        if VERSION >= v"1.7"
            @test @blas(mul!(copy(ba), b, spA, 1, 1)) ≈ b*a + ba atol=atol broken=broken_bXspA
        else
            @test @blas(mul!(copy(ba), b, spA, 1, 1)) ≈ b*a + ba
        end
        @test @blas(mul!(copy(cta), c, transpose(spA), 1, 1)) ≈ c*transpose(a) + cta atol=atol
        @test @blas(mul!(copy(cta), c, spA', 1, 1)) ≈ c*a' + cta atol=atol
    end
end

@testset "SparseTriangular{$T} {* /} Vector{$T}" begin
    for _ in 1:10
        n = rand(50:150)
        spA = sprand(T, n, n, 0.5) + convert(real(T), sqrt(n))*I
        A = Array(spA)
        b = rand(T, n)
        symA = spA + transpose(spA)
        trilA = tril(spA)
        triuA = triu(spA)
        trilUA = tril(spA, -1) + I
        triuUA = triu(spA, 1)  + I

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
end

end
