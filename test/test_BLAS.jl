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
    @test MKLSparse.matdescra(sA) == "GFNF"
end

@testset "convert(MKLSparse.matrix_descr, matdescr::AbstractString)" begin
    @test convert(MKLSparse.matrix_descr, "SLNF") == MKLSparse.matrix_descr(
        MKLSparse.SPARSE_MATRIX_TYPE_SYMMETRIC, MKLSparse.SPARSE_FILL_MODE_LOWER, MKLSparse.SPARSE_DIAG_NON_UNIT)
    @test convert(MKLSparse.matrix_descr, "SUNF") == MKLSparse.matrix_descr(
            MKLSparse.SPARSE_MATRIX_TYPE_SYMMETRIC, MKLSparse.SPARSE_FILL_MODE_UPPER, MKLSparse.SPARSE_DIAG_NON_UNIT)
    @test convert(MKLSparse.matrix_descr, "TLNF") == MKLSparse.matrix_descr(
        MKLSparse.SPARSE_MATRIX_TYPE_TRIANGULAR, MKLSparse.SPARSE_FILL_MODE_LOWER, MKLSparse.SPARSE_DIAG_NON_UNIT)
    @test convert(MKLSparse.matrix_descr, "TUNF") == MKLSparse.matrix_descr(
        MKLSparse.SPARSE_MATRIX_TYPE_TRIANGULAR, MKLSparse.SPARSE_FILL_MODE_UPPER, MKLSparse.SPARSE_DIAG_NON_UNIT)
    @test convert(MKLSparse.matrix_descr, "TLUF") == MKLSparse.matrix_descr(
        MKLSparse.SPARSE_MATRIX_TYPE_TRIANGULAR, MKLSparse.SPARSE_FILL_MODE_LOWER, MKLSparse.SPARSE_DIAG_UNIT)
    @test convert(MKLSparse.matrix_descr, "TUUF") == MKLSparse.matrix_descr(
        MKLSparse.SPARSE_MATRIX_TYPE_TRIANGULAR, MKLSparse.SPARSE_FILL_MODE_UPPER, MKLSparse.SPARSE_DIAG_UNIT)
    @test convert(MKLSparse.matrix_descr, "GFNF") == MKLSparse.matrix_descr(
        MKLSparse.SPARSE_MATRIX_TYPE_GENERAL, MKLSparse.SPARSE_FILL_MODE_FULL, MKLSparse.SPARSE_DIAG_NON_UNIT)
end

# evaluates ex and checks whether it has called any SparseBLAS MKL method
macro blas(ex)
    quote
        begin
            MKLSparse.__mklsparse_calls_count[] = 0
            local res = $(@inferred(esc(ex)))
            @test MKLSparse.__mklsparse_calls_count[] == 1
            res
        end
    end
end

# generate random sparse matrix of the specified type SPMT

function sparserand(::Type{SPMT}, m::Integer, n::Integer, p::Real, diag::Real = 0) where {SPMT <: SparseMatrixCSC{Tv, Ti}} where {Tv, Ti}
    spM = sprand(Tv, m, n, p)
    if diag != 0
        spM += convert(real(Tv), diag)*I
    end
    return convert(SPMT, spM)
end

sparserand(::Type{SPMT}, m::Integer, n::Integer, p::Real, diag::Real = 0) where {SPMT <: MKLSparse.SparseMatrixCSR{Tv, Ti}} where {Tv, Ti} =
    convert(MKLSparse.SparseMatrixCSR, transpose(sparserand(SparseMatrixCSC{Tv, Ti}, n, m, p, diag)))

sparserand(::Type{SPMT}, m::Integer, n::Integer, p::Real, diag::Real = 0) where {SPMT <: MKLSparse.SparseMatrixCOO{Tv, Ti}} where {Tv, Ti} =
    convert(MKLSparse.SparseMatrixCOO, sparserand(SparseMatrixCSC{Tv, Ti}, m, n, p, diag))

# special matrix classes to test
# and the function to create a matrix of that class from a random sparse matrix
matrix_classes = [
    Symmetric => sp -> sp + transpose(sp),
    Hermitian => sp -> sp + adjoint(sp),
    LowerTriangular => sp -> tril(sp),
    UpperTriangular => sp -> triu(sp),
    UnitLowerTriangular => sp -> tril(sp, -1) + I,
    UnitUpperTriangular => sp -> triu(sp, 1) + I,
]

@testset "BLAS for $SPMT{$T, $IT} matrices" for
    SPMT in (SparseMatrixCSC, MKLSparse.SparseMatrixCOO, MKLSparse.SparseMatrixCSR),
    T in (Float32, Float64, ComplexF32, ComplexF64),
    IT in (Base.USE_BLAS64 ? (Int32, Int64) : (Int32,))

local isCOO = SPMT <: MKLSparse.SparseMatrixCOO
local atol::real(T) = 750*eps(real(one(T))) # absolute tolerance for SparseBLAS results

@testset "Create MKLSparse matrix from $SPMT{$T, $IT} and export back" begin
    spA = sparserand(SPMT{T, IT}, rand(10:50), rand(10:50), 0.25)
    mklA = MKLSparse.MKLSparseMatrix(spA)

    # test conversion to incompatible Julia types
    SPMT2 = SPMT === SparseMatrixCSC ? MKLSparse.SparseMatrixCSR : SparseMatrixCSC
    @test_throws MKLSparseError convert(SPMT2{T, IT}, mklA)

    # MKL Sparse does not check for matrix index type and function name compatibility
    #if Base.USE_BLAS64
    #    IT2 = IT === Int64 ? Int32 : Int64
    #    @test_throws MKLSparseError convert(SPMT{T, IT2}, mklA)
    #end
    # MKL Sparse does not check for matrix value type and function name compatibility
    #T2 = T === Float64 ? Float32 : Float64
    #@test_throws MKLSparseError convert(SPMT{T2, IT}, mklA)

    # MKL does not support export of COO matrices
    @test convert(SPMT{T, IT}, mklA) == spA skip=isCOO
end

@testset "$SPMT{$T,$IT} * Vector{$T}" begin
    for _ in 1:10
        m, n = rand(10:50, 2)
        spA = sparserand(SPMT{T, IT}, m, n, 0.5)
        a = convert(Array, spA)
        b = rand(T, n)
        c = rand(T, m)

        @test @blas(spA*b) ≈ a*b atol=atol
        @test @blas(spA'*c) ≈ a'*c atol=atol
        @test @blas(transpose(spA)*c) ≈ transpose(a)*c atol=atol

        if m != n
            @test_throws DimensionMismatch spA*c
            @test_throws DimensionMismatch spA'*b
            @test_throws DimensionMismatch transpose(spA)*b
        end
    end
end

@testset "$trans(Vector{$T}) * $SPMT{$T,$IT}" for trans in (transpose, adjoint)
    for _ in 1:10
        m, n = rand(10:50, 2)
        spA = sparserand(SPMT{T, IT}, m, n, 0.5)
        a = convert(Array, spA)
        b = rand(T, m)
        c = rand(T, n)

        @test @blas(trans(b)*spA) ≈ trans(b)*a atol=atol
        @test @blas(trans(c)*spA') ≈ trans(c)*a' atol=atol
        @test @blas(trans(c)*transpose(spA)) ≈ trans(c)*transpose(a) atol=atol

        if m != n
            @test_throws DimensionMismatch c*spA
            @test_throws DimensionMismatch b*spA'
            @test_throws DimensionMismatch b*transpose(spA)

            @test_throws DimensionMismatch trans(c)*spA
            @test_throws DimensionMismatch trans(b)*spA'
            @test_throws DimensionMismatch trans(b)*transpose(spA)
        end
    end
end

@testset "$SPMT{$T,$IT} * Matrix{$T}" begin
    for _ in 1:10
        m, n, k, l = rand(10:50, 4)
        spA = sparserand(SPMT{T,IT}, m, n, 0.5)
        a = convert(Array, spA)
        b = rand(T, n, k)
        c = rand(T, m, l)
        ab = rand(T, m, k)
        tac = rand(T, n, l)
        α, β = rand(T, 2)

        @test @blas(spA*b) ≈ a*b atol=atol
        @test @blas(spA'*c) ≈ a'*c atol=atol
        @test @blas(transpose(spA)*c) ≈ transpose(a)*c atol=atol

        if m != n
            @test_throws DimensionMismatch spA*c
            @test_throws DimensionMismatch spA'*b
            @test_throws DimensionMismatch transpose(spA)*b
        end

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

if SPMT <: SparseMatrixCSC # conversion to special matrices not implemented for CSR and COO

@testset "$Aclass{$SPMT{$T}} * $trans($(ifelse(Bdim == 2, "Matrix", "Vector")){$T})" for Bdim in 1:2,
        (Aclass, convert_to_class) in matrix_classes,
        trans in (identity, transpose, adjoint)

    (Bdim == 1 && trans != identity) && continue # not valid

    for _ in 1:10
        n = rand(50:150)
        spA = convert_to_class(sparserand(SPMT{T,IT}, n, n, 0.5, sqrt(n)))
        A = convert(Array, spA)
        @test spA == A
        B = Bdim == 2 ? rand(T, n, n) : rand(T, n)
        α = rand(T)

        @test @blas(mul!(similar(B), Aclass(spA), trans(B), α, 0)) ≈ α * Aclass(A) * trans(B) skip=(Bdim==2 && trans!=identity)
        @test @blas(mul!(similar(B), Aclass(spA), trans(B))) ≈ Aclass(A) * trans(B) skip=(Bdim==2 && trans!=identity)
        @test @blas(Aclass(spA) * trans(B)) ≈ Aclass(A) * trans(B) skip=(Bdim==2 && trans!=identity)
    end
end

@testset "$Aclass{$SPMT{$T}} \\ $(ifelse(Bdim == 2, "Matrix", "Vector")){$T}" for Bdim in 1:2,
    (Aclass, convert_to_class) in matrix_classes

    (Aclass == Symmetric || Aclass == Hermitian) && continue # not implemented in MKLSparse

    for _ in 1:10
        n = rand(50:150)
        spA = convert_to_class(sparserand(SPMT{T,IT}, n, n, 0.5, sqrt(n)))
        A = convert(Array, spA)
        B = Bdim == 2 ? rand(T, n, rand(50:150)) : rand(T, n)
        spAclass = Aclass(spA)
        α = rand(T)

        @test @blas(ldiv!(similar(B), Aclass(spA), B, α)) ≈ α * (Aclass(A) \ B)
        @test @blas(ldiv!(similar(B), Aclass(spA), B)) ≈ Aclass(A) \ B
        @test @blas(Aclass(spA) \ B) ≈ Aclass(A) \ B
    end
end

end

end

