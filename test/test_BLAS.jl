using MKLSparse
using Test, SparseArrays, LinearAlgebra

ntries = 10 # how many random matrices to test
max_el = 3  # maximal absolute value of matrix element

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

_clamp(x, min, max) = clamp(x, min, max)
_clamp(x::Complex, min::Real, max::Real) =
    typeof(x)(clamp(real(x), min, max), clamp(imag(x), min, max))

# generate random sparse matrix of the specified type SPMT

function sparserandn(::Type{SPMT}, m::Integer, n::Integer, p::Real, diag::Real = 0;
                     max_el::Real = max_el) where {SPMT <: SparseMatrixCSC{Tv, Ti}} where {Tv, Ti}
    spM = sprandn(Tv, m, n, p)
    if diag != 0
        spM += convert(real(Tv), diag)*I
    end
    nzM = nonzeros(spM)
    @inbounds for i in eachindex(nzM)
        nzM[i] = _clamp(nzM[i], -max_el, max_el)
    end
    return convert(SPMT, spM)
end

function denserandn(::Type{Tv}, sz...; max_el::Real = max_el) where {Tv}
    M = randn(Tv, sz...)
    if M isa AbstractArray
        @inbounds for i in eachindex(M)
            M[i] = _clamp(M[i], -max_el, max_el)
        end
    else
        M = _clamp(M, -max_el, max_el)
    end
    return M
end

sparserandn(::Type{SPMT}, m::Integer, n::Integer, p::Real, diag::Real = 0) where {SPMT <: MKLSparse.SparseMatrixCSR{Tv, Ti}} where {Tv, Ti} =
    convert(MKLSparse.SparseMatrixCSR, transpose(sparserandn(SparseMatrixCSC{Tv, Ti}, n, m, p, diag)))

sparserandn(::Type{SPMT}, m::Integer, n::Integer, p::Real, diag::Real = 0) where {SPMT <: MKLSparse.SparseMatrixCOO{Tv, Ti}} where {Tv, Ti} =
    convert(MKLSparse.SparseMatrixCOO, sparserandn(SparseMatrixCSC{Tv, Ti}, m, n, p, diag))

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
    SPMT in (SparseMatrixCSC, MKLSparse.SparseMatrixCSR, MKLSparse.SparseMatrixCOO),
    T in (Float32, Float64, ComplexF32, ComplexF64),
    IT in (Base.USE_BLAS64 ? (Int32, Int64) : (Int32,))

local isCOO = SPMT <: MKLSparse.SparseMatrixCOO
local isCOOorCSR = isCOO || SPMT <: MKLSparse.SparseMatrixCSR
local atol::real(T) = 750*eps(real(one(T))) # absolute tolerance for SparseBLAS results

@testset "Describe $SPMT{$T, $IT} matrix" begin
    spA = sparserandn(SPMT{T, IT}, 10, 10, 0.25)

    @test MKLSparse.describe_and_unwrap(spA) == ('N', MKLSparse.matrix_descr('G','F','N'), spA)
    @test MKLSparse.describe_and_unwrap(transpose(spA)) == ('T', MKLSparse.matrix_descr('G','F','N'), spA)
    @test MKLSparse.describe_and_unwrap(adjoint(spA)) == ('C', MKLSparse.matrix_descr('G','F','N'), spA)

    @test MKLSparse.describe_and_unwrap(Symmetric(spA)) == ('N', MKLSparse.matrix_descr('S','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(Hermitian(spA)) == ('N', MKLSparse.matrix_descr('H','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(LowerTriangular(spA)) == ('N', MKLSparse.matrix_descr('T','L','N'), spA)
    @test MKLSparse.describe_and_unwrap(UpperTriangular(spA)) == ('N', MKLSparse.matrix_descr('T','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(UnitLowerTriangular(spA)) == ('N', MKLSparse.matrix_descr('T','L','U'), spA)
    @test MKLSparse.describe_and_unwrap(UnitUpperTriangular(spA)) == ('N', MKLSparse.matrix_descr('T','U','U'), spA)

    @test MKLSparse.describe_and_unwrap(adjoint(Symmetric(spA))) == (T <: Real ? 'N' : 'C', MKLSparse.matrix_descr('S','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(adjoint(Hermitian(spA))) == ('N', MKLSparse.matrix_descr('H','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(adjoint(LowerTriangular(spA))) == ('C', MKLSparse.matrix_descr('T','L','N'), spA)
    @test MKLSparse.describe_and_unwrap(adjoint(UpperTriangular(spA))) == ('C', MKLSparse.matrix_descr('T','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(adjoint(UnitLowerTriangular(spA))) == ('C', MKLSparse.matrix_descr('T','L','U'), spA)
    @test MKLSparse.describe_and_unwrap(adjoint(UnitUpperTriangular(spA))) == ('C', MKLSparse.matrix_descr('T','U','U'), spA)

    @test MKLSparse.describe_and_unwrap(transpose(Symmetric(spA))) == ('N', MKLSparse.matrix_descr('S','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(transpose(Hermitian(spA))) == (T <: Real ? 'N' : 'T', MKLSparse.matrix_descr('H','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(transpose(LowerTriangular(spA))) == ('T', MKLSparse.matrix_descr('T','L','N'), spA)
    @test MKLSparse.describe_and_unwrap(transpose(UpperTriangular(spA))) == ('T', MKLSparse.matrix_descr('T','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(transpose(UnitLowerTriangular(spA))) == ('T', MKLSparse.matrix_descr('T','L','U'), spA)
    @test MKLSparse.describe_and_unwrap(transpose(UnitUpperTriangular(spA))) == ('T', MKLSparse.matrix_descr('T','U','U'), spA)

    @test MKLSparse.describe_and_unwrap(Symmetric(adjoint(spA))) == (T <: Real ? 'N' : 'C', MKLSparse.matrix_descr('S','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(Hermitian(adjoint(spA))) == ('N', MKLSparse.matrix_descr('H','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(UpperTriangular(adjoint(spA))) == ('C', MKLSparse.matrix_descr('T','L','N'), spA)
    @test MKLSparse.describe_and_unwrap(LowerTriangular(adjoint(spA))) == ('C', MKLSparse.matrix_descr('T','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(UnitUpperTriangular(adjoint(spA))) == ('C', MKLSparse.matrix_descr('T','L','U'), spA)
    @test MKLSparse.describe_and_unwrap(UnitLowerTriangular(adjoint(spA))) == ('C', MKLSparse.matrix_descr('T','U','U'), spA)

    @test MKLSparse.describe_and_unwrap(Symmetric(transpose(spA))) == ('N', MKLSparse.matrix_descr('S','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(Hermitian(transpose(spA))) == (T <: Real ? 'N' : 'T', MKLSparse.matrix_descr('H','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(UpperTriangular(transpose(spA))) == ('T', MKLSparse.matrix_descr('T','L','N'), spA)
    @test MKLSparse.describe_and_unwrap(LowerTriangular(transpose(spA))) == ('T', MKLSparse.matrix_descr('T','U','N'), spA)
    @test MKLSparse.describe_and_unwrap(UnitUpperTriangular(transpose(spA))) == ('T', MKLSparse.matrix_descr('T','L','U'), spA)
    @test MKLSparse.describe_and_unwrap(UnitLowerTriangular(transpose(spA))) == ('T', MKLSparse.matrix_descr('T','U','U'), spA)
end

@testset "Create MKLSparse matrix from $SPMT{$T, $IT} and export back" begin
    spA = sparserandn(SPMT{T, IT}, rand(10:50), rand(10:50), 0.25)
    mklA = MKLSparse.MKLSparseMatrix(spA)

    # test conversion to incompatible Julia types
    SPMT2 = SPMT === SparseMatrixCSC ? MKLSparse.SparseMatrixCSR : SparseMatrixCSC
    @test_throws MethodError convert(SPMT2{T, IT}, mklA)

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
    MKLSparse.destroy(mklA)
end

@testset "$SPMT{$T,$IT} * Vector{$T}" begin
    for _ in 1:ntries
        m, n = rand(10:50, 2)
        spA = sparserandn(SPMT{T, IT}, m, n, 0.5)
        a = convert(Array, spA)
        b = denserandn(T, n)
        c = denserandn(T, m)

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
    for _ in 1:ntries
        m, n = rand(10:50, 2)
        spA = sparserandn(SPMT{T, IT}, m, n, 0.5)
        a = convert(Array, spA)
        b = denserandn(T, m)
        c = denserandn(T, n)

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
    for _ in 1:ntries
        m, n, k, l = rand(10:50, 4)
        spA = sparserandn(SPMT{T,IT}, m, n, 0.5)
        a = convert(Array, spA)
        b = denserandn(T, n, k)
        c = denserandn(T, m, l)
        ab = denserandn(T, m, k)
        tac = denserandn(T, n, l)
        α, β = denserandn(T, 2)

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

@testset "Matrix{$T} * $SPMT{$T,$IT}" begin
    for _ in 1:ntries
        m, n, k, l = rand(10:50, 4)
        spA = sparserandn(SPMT{T,IT}, m, n, 0.5)
        a = convert(Array, spA)
        b = denserandn(T, k, m)
        c = denserandn(T, l, n)
        ba = denserandn(T, k, n)
        cta = denserandn(T, l, m)
        α, β = denserandn(T, 2)

        # COO and CSR is currently not supported
        # (MKLSparse does not support this combination of indexing, sparse and dense layouts)
        @test @blas(b*spA) ≈ b*a atol=atol skip=isCOOorCSR
        @test @blas(c*spA') ≈ c*a' atol=atol skip=isCOOorCSR
        @test @blas(c*transpose(spA)) ≈ c*transpose(a) atol=atol skip=isCOOorCSR

        if m != n
            @test_throws DimensionMismatch c*spA
            @test_throws DimensionMismatch b*spA'
            @test_throws DimensionMismatch b*transpose(spA)
        end

        @test @blas(mul!(similar(ba), b, spA)) ≈ b*a atol=atol skip=isCOOorCSR
        @test @blas(mul!(similar(cta), c, spA')) ≈ c*a' atol=atol skip=isCOOorCSR
        @test @blas(mul!(similar(cta), c, transpose(spA))) ≈ c*transpose(a) atol=atol skip=isCOOorCSR

        @test @blas(mul!(copy(ba), b, spA, α, β)) ≈ α*b*a + β*ba atol=atol skip=isCOOorCSR
        @test @blas(mul!(copy(cta), c, spA', α, β)) ≈ α*c*a' + β*cta atol=atol skip=isCOOorCSR
        @test @blas(mul!(copy(cta), c, transpose(spA), α, β)) ≈ α*c*transpose(a) + β*cta atol=atol skip=isCOOorCSR

        @test @blas(mul!(copy(ba), b, spA, 1, 1)) ≈ b*a + ba atol=atol skip=isCOOorCSR
        @test @blas(mul!(copy(cta), c, transpose(spA), 1, 1)) ≈ c*transpose(a) + cta atol=atol skip=isCOOorCSR
        @test @blas(mul!(copy(cta), c, spA', 1, 1)) ≈ c*a' + cta atol=atol skip=isCOOorCSR
    end
end

if SPMT <: SparseMatrixCSC # conversion to special matrices not implemented for CSR and COO

@testset "$Aclass{$SPMT{$T}} * $trans($(ifelse(Bdim == 2, "Matrix", "Vector")){$T})" for Bdim in 1:2,
        (Aclass, convert_to_class) in matrix_classes,
        trans in (identity, transpose, adjoint)

    (Bdim == 1 && trans != identity) && continue # not valid

    for _ in 1:ntries
        n = rand(50:150)
        spf = 0.1 + 0.8 * rand()
        spA = convert_to_class(sparserandn(SPMT{T,IT}, n, n, spf, sqrt(n)))
        A = convert(Array, spA)
        @test spA == A
        B = Bdim == 2 ? denserandn(T, n, n) : denserandn(T, n)
        C = denserandn(T, size(B))
        α, β = denserandn(T, 2)

        @test @blas(mul!(copy(C), Aclass(spA), trans(B), α, β)) ≈ mul!(copy(C), Aclass(A), trans(B), α, β) skip=(Bdim==2 && trans!=identity)
        @test @blas(mul!(copy(C), Aclass(spA), trans(B))) ≈ Aclass(A) * trans(B) skip=(Bdim==2 && trans!=identity)
        @test @blas(Aclass(spA) * trans(B)) ≈ Aclass(A) * trans(B) skip=(Bdim==2 && trans!=identity)
    end
end

@testset "$trans($(ifelse(Bdim == 2, "Matrix", "Vector")){$T}) * $Aclass{$SPMT{$T}}" for Bdim in 1:2,
        (Aclass, convert_to_class) in matrix_classes,
        trans in (identity, transpose, adjoint)

    (Bdim == 1 && trans == identity) && continue # not valid

    for _ in 1:ntries
        n = rand(50:150)
        spA = convert_to_class(sparserandn(SPMT{T,IT}, n, n, 0.5, sqrt(n)))
        A = convert(Array, spA)
        @test spA == A
        B = Bdim == 2 ? denserandn(T, n, n) : denserandn(T, n)
        α = denserandn(T)

        @test @blas(mul!(similar(B), trans(B), Aclass(spA), α, 0)) ≈ α * trans(B) * Aclass(A) skip=trans!=identity
        @test @blas(mul!(similar(B), trans(B), Aclass(spA))) ≈ trans(B) * Aclass(A) skip=trans!=identity
        @test @blas(trans(B) * Aclass(spA)) ≈ trans(B) * Aclass(A) skip=(Bdim==2 && trans!=identity)
    end
end

@testset "$Aclass{$SPMT{$T}} \\ $(ifelse(Bdim == 2, "Matrix", "Vector")){$T}" for Bdim in 1:2,
    (Aclass, convert_to_class) in matrix_classes

    (Aclass == Symmetric || Aclass == Hermitian) && continue # not implemented in MKLSparse

    for _ in 1:ntries
        n = rand(50:150)
        spA = convert_to_class(sparserandn(SPMT{T,IT}, n, n, 0.5, sqrt(n)))
        A = convert(Array, spA)
        B = Bdim == 2 ? denserandn(T, n, rand(50:150)) : denserandn(T, n)
        spAclass = Aclass(spA)
        α = denserandn(T)

        @test @blas(ldiv!(similar(B), Aclass(spA), B, α)) ≈ α * (Aclass(A) \ B)
        @test @blas(ldiv!(similar(B), Aclass(spA), B)) ≈ Aclass(A) \ B
        @test @blas(Aclass(spA) \ B) ≈ Aclass(A) \ B
    end
end

end

end

