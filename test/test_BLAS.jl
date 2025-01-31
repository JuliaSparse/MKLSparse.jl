using MKLSparse
using Test, SparseArrays, LinearAlgebra

ntries = 10 # how many random matrices to test
max_el = 3  # maximal absolute value of matrix element

@testset "MKLSparse.matdescra()" begin
    sA = sprand(5, 5, 0.1)
    sS = sA' + sA
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

@testset "fastcopytri!(Matrix{$T})" for
    T in (Float32, Float64, ComplexF32, ComplexF64)

    for _ in 1:ntries
        m, n = rand(10:100, 2)

        if m != n
            @test_throws DimensionMismatch MKLSparse.fastcopytri!(denserandn(T, m, n), 'L')
        end

        A = denserandn(T, m, m)
        @test_throws ArgumentError MKLSparse.fastcopytri!(A, 'K')

        AL = copy(A)
        @test MKLSparse.fastcopytri!(AL, 'L') === AL

        @test MKLSparse.fastcopytri!(copy(A), 'L') == LinearAlgebra.copytri!(copy(A), 'L')
        @test MKLSparse.fastcopytri!(copy(A), 'U') == LinearAlgebra.copytri!(copy(A), 'U')

        @test MKLSparse.fastcopytri!(copy(A), 'L', true) == LinearAlgebra.copytri!(copy(A), 'L', true)
        @test MKLSparse.fastcopytri!(copy(A), 'U', true) == LinearAlgebra.copytri!(copy(A), 'U', true)

        @test MKLSparse.fastcopytri!(copy(A), 'L', false) == LinearAlgebra.copytri!(copy(A), 'L', false)
        @test MKLSparse.fastcopytri!(copy(A), 'U', false) == LinearAlgebra.copytri!(copy(A), 'U', false)
    end
end

@testset "{$T, $IT} Sparse Matrices conversion" for
    T in (Float32, Float64, ComplexF32, ComplexF64),
    IT in (Base.USE_BLAS64 ? (Int32, Int64) : (Int32,))

    @testset "$SPMT{$T, $IT}" for
        SPMT in (SparseMatrixCSC, MKLSparse.SparseMatrixCSR,
                 MKLSparse.SparseMatrixCOO)
        for _ in 1:ntries
            m, n = rand(10:50, 4)
            spf = 0.1 + 0.8 * rand()
            spA = sparserandn(SPMT{T,IT}, m, n, spf)

            A = convert(Matrix, spA)
            @test A isa Matrix{T}
            @test size(A) == size(spA)
            @test nnz(spA) == sum(!=(0), A)

            AA = convert(Array, spA)
            @test AA isa Matrix{T}
            @test AA == A
        end
    end

    for _ in 1:ntries
        m, n = rand(10:50, 4)
        spf = 0.1 + 0.8 * rand()
        cscA = sparserandn(SparseMatrixCSC{T,IT}, m, n, spf)

        csrA = convert(MKLSparse.SparseMatrixCSR{T,IT}, transpose(cscA))
        @test csrA isa MKLSparse.SparseMatrixCSR{T,IT}
        @test size(cscA) == reverse(size(csrA))
        @test convert(Matrix, transpose(cscA)) == convert(Matrix, csrA)

        csrB = sparserandn(MKLSparse.SparseMatrixCSR{T,IT}, m, n, spf)
        cscB = convert(SparseMatrixCSC{T,IT}, transpose(csrB))
        @test cscB isa SparseMatrixCSC{T,IT}
        @test size(csrB) == reverse(size(cscB))
        @test convert(Matrix, csrB) == convert(Matrix, transpose(cscB))

        cooA = convert(MKLSparse.SparseMatrixCOO{T,IT}, cscA)
        @test cooA isa MKLSparse.SparseMatrixCOO{T,IT}
        @test size(cscA) == size(cooA)
        @test convert(Matrix, cscA) == convert(Matrix, cooA)
    end
end

@testset "BLAS for $SPMT{$T, $IT} matrices" for
    SPMT in (SparseMatrixCSC, MKLSparse.SparseMatrixCSR, MKLSparse.SparseMatrixCOO),
    T in (Float32, Float64, ComplexF32, ComplexF64),
    IT in (Base.USE_BLAS64 ? (Int32, Int64) : (Int32,))

local isCOO = SPMT <: MKLSparse.SparseMatrixCOO
local isCOOorCSR = isCOO || SPMT <: MKLSparse.SparseMatrixCSR
local atol::real(T) = ifelse(T <: Complex, 1E+3, 1E+3) * eps(real(one(T))) # absolute tolerance for SparseBLAS results

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

if SPMT <: Union{SparseMatrixCSC, MKLSparse.SparseMatrixCSR}

# dense := sparse * sparse for generic matrices
@testset "mul!(dense, $SPMT{$T}, $SPMT{$T})" begin
    for _ in 1:ntries
        m, n, k, l = rand(10:50, 4)
        spf = 0.1 + 0.8 * rand()
        spA = sparserandn(SPMT{T,IT}, m, n, spf)
        spB = sparserandn(SPMT{T,IT}, n, k, spf)
        spC = sparserandn(SPMT{T,IT}, m, l, spf)
        spD = sparserandn(SPMT{T,IT}, k, n, spf)
        spE = sparserandn(SPMT{T,IT}, l, m, spf)
        A = convert(Array, spA)
        B = convert(Array, spB)
        C = convert(Array, spC)
        D = convert(Array, spD)
        E = convert(Array, spE)
        AB = denserandn(T, m, k)
        tAC = denserandn(T, n, l)
        AtA = denserandn(T, m, m)
        tAA = denserandn(T, n, n)
        α, β = denserandn(T, 2)

        if m != n
            @test_throws DimensionMismatch mul!(denserandn(T, m, l), spA, spC)
            @test_throws DimensionMismatch mul!(denserandn(T, n, n), transpose(spA), spB)
            @test_throws DimensionMismatch mul!(denserandn(T, n, l), spA, spB)
        end

        @test @blas(mul!(randn(T, m, k), spA, spB)) ≈ A * B atol = atol
        @test @blas(mul!(copy(AB), spA, spB, α, β)) ≈ α * A * B + β * AB atol = atol
        @test @blas(mul!(copy(AB), spA, spB, 1, 1)) ≈ A * B + AB atol = atol

        for trans in (transpose, adjoint)
            if (T <: Complex) && (trans == adjoint)
                # adjoint of complex matrices seems to be not implemented
                @test_throws MKLSparseError mul!(similar(tAC), trans(spA), spC)
                continue
            end

            @test @blas(mul!(randn(T, n, l), trans(spA), spC)) ≈ trans(A) * C atol = atol
            @test @blas(mul!(randn(T, m, k), spA, trans(spD))) ≈ A * trans(D) atol = atol
            @test @blas(mul!(randn(T, n, l), trans(spA), trans(spE))) ≈ trans(A) * trans(E) atol = atol

            @test @blas(mul!(copy(tAC), trans(spA), spC, α, β)) ≈ α * trans(A) * C + β * tAC atol = atol
            @test @blas(mul!(copy(AB), spA, trans(spD), α, β)) ≈ α * A * trans(D) + β * AB atol = atol
            @test @blas(mul!(copy(tAC), trans(spA), trans(spE), α, β)) ≈ α * trans(A) * trans(E) + β * tAC atol = atol

            @test @blas(mul!(copy(tAC), trans(spA), spC, 1, 1)) ≈ trans(A) * C + tAC atol = atol
            @test @blas(mul!(copy(AB), spA, trans(spD), 1, 1)) ≈ A * trans(D) + AB atol = atol
            @test @blas(mul!(copy(tAC), trans(spA), trans(spE), 1, 1)) ≈ trans(A) * trans(E) + tAC atol = atol

            # matrix multiplication with itself (3-arg version uses syrk)
            @test @blas(mul!(copy(AtA), spA, trans(spA))) ≈ A * trans(A) atol = atol
            @test @blas(mul!(copy(tAA), trans(spA), spA)) ≈ trans(A) * A atol = atol
            @test @blas(mul!(copy(AtA), spA, trans(spA), α, β)) ≈ α * A * trans(A) + β * AtA atol = atol
            @test @blas(mul!(copy(tAA), trans(spA), spA, α, β)) ≈ α * trans(A) * A + β * tAA atol = atol
        end
    end
end

# sparse := sparse * sparse for generic matrices
@testset "$SPMT{$T}: C := A * B and mul!(C, A, B)" begin
    @testset "empty result" begin
        A = convert(SPMT, sparse(IT[1], IT[2], ones(T, 1), 2, 2))
        B = A * A
        @test convert(Matrix, B) == zeros(T, 2, 2)
        # in-place multiplication of the empty matrix
        mul!(convert(SPMT, spzeros(T, IT, 2, 2)), A, A)
    end

    for _ in 1:ntries
        m, n, k, l = rand(10:50, 4)
        spf = 0.1 + 0.8 * rand()
        spA = sparserandn(SPMT{T,IT}, m, n, spf)
        spB = sparserandn(SPMT{T,IT}, n, k, spf)
        spC = sparserandn(SPMT{T,IT}, m, l, spf)
        spD = sparserandn(SPMT{T,IT}, k, n, spf)
        spE = sparserandn(SPMT{T,IT}, l, m, spf)
        A = convert(Matrix, spA)
        B = convert(Matrix, spB)
        C = convert(Matrix, spC)
        D = convert(Matrix, spD)
        E = convert(Matrix, spE)
        # sparse result of the spA*spB
        # cannot convert dense into sparse since it generates sorted rowvals,
        # while MKLSparse generates non-sorted rowvals
        # keep the sparsity pattern, but randomize the nzvalues
        spAB = convert(SPMT{T,IT}, spA * spB)
        nonzeros(spAB) .= randn(T, nnz(spAB))
        sptAC = convert(SPMT{T,IT}, transpose(spA) * spC)
        nonzeros(sptAC) .= randn(T, nnz(sptAC))
        spAtD = convert(SPMT{T,IT}, spA * transpose(spD))
        nonzeros(spAtD) .= randn(T, nnz(spAtD))
        sptAtE = convert(SPMT{T,IT}, transpose(spA) * transpose(spE))
        nonzeros(sptAtE) .= randn(T, nnz(sptAtE))
        sptAA = convert(SPMT{T,IT}, transpose(spA) * spA)
        nonzeros(sptAA) .= randn(T, nnz(sptAA))
        spAtA = convert(SPMT{T,IT}, spA * transpose(spA))
        nonzeros(spAtA) .= randn(T, nnz(spAtA))
        α, β = denserandn(T, 2)

        if m != n
            @test_throws DimensionMismatch @blas(spA * spC)
            @test_throws DimensionMismatch @blas(spA' * spB)
            @test_throws DimensionMismatch @blas(transpose(spA) * spB)
            @test_throws DimensionMismatch @blas(mul!(similar(sptAC), spA, spB))
        end

        @test @blas(spA * spB) isa SPMT{T,IT} # test that the result is sparse
        @test convert(Matrix, @blas(spA * spB)) ≈ A * B atol=atol

        @test convert(Matrix, @blas(mul!(copy(spAB), spA, spB))) ≈ A * B atol=atol
        # 5-arg mul!() is not implemented

        # cannot in-place multiply into the sparse matrix with the different sparsity pattern
        @test_throws ErrorException mul!(sparserandn(SPMT{T, IT}, m, k, spf), spA, spB)

        for trans in (transpose, adjoint)
            @test convert(Matrix, @blas(spA * trans(spD))) ≈ A * trans(D) atol=atol
            @test convert(Matrix, @blas(trans(spA) * spC)) ≈ trans(A) * C atol=atol
            @test convert(Matrix, @blas(trans(spA) * trans(spE))) ≈ trans(A) * trans(E) atol=atol

            @test convert(Matrix, @blas(mul!(copy(sptAC), trans(spA), spC))) ≈ trans(A) * C atol=atol
            @test convert(Matrix, @blas(mul!(copy(spAtD), spA, trans(spD)))) ≈ A * trans(D) atol=atol
            @test convert(Matrix, @blas(mul!(copy(sptAtE), trans(spA), trans(spE)))) ≈ trans(A) * trans(E) atol=atol

            # 5-arg mul!() is not implemented

            # matrix multiplication with itself
            @test convert(Matrix, @blas(mul!(copy(spAtA), spA, trans(spA)))) ≈ A * trans(A) atol=atol
            @test convert(Matrix, @blas(mul!(copy(sptAA), trans(spA), spA))) ≈ trans(A) * A atol=atol
        end
    end
end

@testset "A::$SPMT{$T}, Matrix{$T} := A * Aᵗ (syrkd!())" begin
    for _ in 1:ntries
        m, n = rand(10:100, 2)
        spA = sparserandn(SPMT{T, IT}, m, n, 0.1 + 0.8 * rand())
        A = convert(Matrix, spA)
        C1 = denserandn(T, m, m)
        C1 = convert(Matrix{T}, C1 + C1')
        C2 = denserandn(T, n, n)
        C2 = convert(Matrix{T}, C2 + C2')
        # Complex case does not support non-real coefficients due to Hermitian requirement
        α, β = T(real(rand(T))), T(real(rand(T)))

        if T <: Complex && SPMT <: SparseMatrixCSC
            @test_throws "syrkd!() wrapper does not support" @blas(MKLSparse.syrkd!('N', α, spA, β, copy(C2)))
            break
        end

        if m != n
            @test_throws DimensionMismatch @blas(MKLSparse.syrkd!('N', α, spA, β, copy(C2)))
            @test_throws DimensionMismatch @blas(MKLSparse.syrkd!('T', α, spA, β, copy(C1)))
        end

        @test @blas(MKLSparse.syrkd!('N', α, spA, zero(T), copy(C1))) ≈ α * A * A' atol=3*atol
        @test @blas(MKLSparse.syrkd!('T', α, spA, zero(T), copy(C2))) ≈ α * A' * A atol=3*atol
        @test @blas(MKLSparse.syrkd!('C', α, spA, zero(T), copy(C2))) ≈ α * A' * A atol=3*atol

        @test @blas(MKLSparse.syrkd!('N', α, spA, β, copy(C1))) ≈ α * A * A' + β * C1 atol=3*atol
        @test @blas(MKLSparse.syrkd!('T', α, spA, β, copy(C2))) ≈ α * A' * A + β * C2 atol=3*atol
        @test @blas(MKLSparse.syrkd!('C', α, spA, β, copy(C2))) ≈ α * A' * A + β * C2 atol=3*atol
    end
end

@testset "A::$SPMT{$T}, $SPMT{$T} := A * Aᵗ (syrk())" begin
    for _ in 1:ntries
        m, n = rand(10:100, 2)
        spA = sparserandn(SPMT{T, IT}, m, n, 0.1 + 0.8 * rand())
        A = convert(Matrix, spA)

        @test @blas(MKLSparse.syrk('N', spA)) isa SPMT{T}

        # syrk() only returns the upper/lower triangle of the result,
        # copytri!(sparse) is not done since it would change the matrix nonzero structure
        tri = SPMT <: SparseMatrixCSC ? LowerTriangular :
              SPMT <: MKLSparse.SparseMatrixCSR ? UpperTriangular :
              error("unsupported sparse matrix type $SPMT")
        @test convert(Matrix, @blas(MKLSparse.syrk('N', spA))) ≈ tri(A * A') atol=3*atol
        @test convert(Matrix, @blas(MKLSparse.syrk('T', spA))) ≈ tri(A' * A) atol=3*atol
        @test convert(Matrix, @blas(MKLSparse.syrk('C', spA))) ≈ tri(A' * A) atol=3*atol
    end
end

@testset "A::$SPMT{$T}, B:Matrix{$T}: Matrix{$T} := A * B * Aᵗ (syprd!())" begin
    for _ in 1:ntries
        m, n = rand(10:100, 2)
        spA = sparserandn(SPMT{T, IT}, m, n, 0.1 + 0.8 * rand())
        A = convert(Matrix, spA)
        B1 = denserandn(T, n, n)
        B1 = B1 + B1'
        B1 = convert(Matrix{T}, B1 / norm(B1))
        B2 = denserandn(T, m, m)
        B2 = B2 + B2'
        B2 = convert(Matrix{T}, B2 / norm(B2))
        C1 = denserandn(T, m, m)
        C1 = convert(Matrix{T}, C1 + C1')
        C2 = denserandn(T, n, n)
        C2 = convert(Matrix{T}, C2 + C2')
        # Complex case does not support non-real coefficients due to Hermitian requirement
        α, β = T.(real.(denserandn(T, 2)))

        if T <: Complex && SPMT <: SparseMatrixCSC
            @test_throws "syprd!() wrapper does not support" @blas(MKLSparse.syprd!('N', α, spA, B1, zero(T), copy(C1)))
            break
        end

        if m != n
            @test_throws DimensionMismatch @blas(MKLSparse.syprd!('N', α, spA, B1, β, copy(C2)))
            @test_throws DimensionMismatch @blas(MKLSparse.syprd!('N', α, spA, B2, β, copy(C1)))
            @test_throws DimensionMismatch @blas(MKLSparse.syprd!('T', α, spA, B1, β, copy(C2)))
            @test_throws DimensionMismatch @blas(MKLSparse.syprd!('T', α, spA, B2, β, copy(C1)))
        end

        # test with bigger atol since A*B*Aᵗ does more * and + than other routines
        @test @blas(MKLSparse.syprd!('N', α, spA, B1, zero(T), copy(C1))) ≈ α * A * B1 * A' atol=3*atol
        @test @blas(MKLSparse.syprd!('T', α, spA, B2, zero(T), copy(C2))) ≈ α * A' * B2 * A atol=3*atol
        @test @blas(MKLSparse.syprd!('C', α, spA, B2, zero(T), copy(C2))) ≈ α * A' * B2 * A atol=3*atol

        @test @blas(MKLSparse.syprd!('N', α, spA, B1, β, copy(C1))) ≈ α * A * B1 * A' + β * C1 atol=3*atol
        @test @blas(MKLSparse.syprd!('T', α, spA, B2, β, copy(C2))) ≈ α * A' * B2 * A + β * C2 atol=3*atol
        @test @blas(MKLSparse.syprd!('C', α, spA, B2, β, copy(C2))) ≈ α * A' * B2 * A + β * C2 atol=3*atol
    end
end

else # COO

@testset "A,B::$SPMT{$T}: A * B not supported" begin
    m, n, k, l = rand(10:50, 4)
    spf = 0.1 + 0.8 * rand()
    spA = sparserandn(SPMT{T,IT}, m, n, spf)
    spB = sparserandn(SPMT{T,IT}, n, k, spf)
    AB = denserandn(T, m, k)
    α, β = denserandn(T, 2)

    @test_throws MKLSparseError @blas(mul!(AB, spA, spB, α, β))
    @test_throws MKLSparseError @blas(spA * spB)
end

@testset "A::$SPMT{$T}: syrkd!(), syrk(), syprd!() not supported" begin
    m, n = rand(10:100, 2)
    spA = sparserandn(SPMT{T, IT}, m, n, 0.1 + 0.8 * rand())
    A = convert(Matrix, spA)
    B1 = denserandn(T, n, n)
    B2 = denserandn(T, m, m)
    C1 = denserandn(T, m, m)
    α, β = denserandn(T, 2)

    @test_throws MethodError @blas(MKLSparse.syrkd!('N', α, spA, β, copy(C1)))
    @test_throws MethodError @blas(MKLSparse.syrk('N', spA))
    @test_throws MethodError @blas(MKLSparse.syprd!('N', α, spA, B1, β, copy(C1)))
end

end # COO

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

# dense := sparse * sparse for special matrices
@testset "mul!(dense, $Aclass{$SPMT{$T}}, $Bclass{$SPMT{$T}})" for
        (Aclass, convert_to_Aclass) in matrix_classes,
        (Bclass, convert_to_Bclass) in matrix_classes

    # Tests for dense := sparse * sparse (sp2md!())
    for _ in 1:ntries
        n = rand(50:150)
        spf = 0.1 + 0.8 * rand()
        spA = convert_to_Aclass(sparserandn(SPMT{T, IT}, n, n, spf))
        spB = convert_to_Bclass(sparserandn(SPMT{T, IT}, n, n, spf))
        A = convert(Matrix, spA)
        B = convert(Matrix, spB)
        C = denserandn(T, (n, n))
        α, β = denserandn(T, 2)

        # 5-arg
        @test @blas(mul!(copy(C), Aclass(spA), Bclass(spB), α, β)) ≈ mul!(copy(C), A, B, α, β)

        @test @blas(mul!(copy(C), transpose(Aclass(spA)), Bclass(spB), α, β)) ≈ mul!(copy(C), transpose(A), B, α, β)
        @test @blas(mul!(copy(C), Aclass(spA), transpose(Bclass(spB)), α, β)) ≈ mul!(copy(C), A, transpose(B), α, β)
        @test @blas(mul!(copy(C), transpose(Aclass(spA)), transpose(Bclass(spB)), α, β)) ≈ mul!(copy(C), transpose(A), transpose(B), α, β)

        # 3-arg
        @test @blas(mul!(copy(C), Aclass(spA), Bclass(spB))) ≈ A * B

        @test @blas(mul!(copy(C), transpose(Aclass(spA)), Bclass(spB))) ≈ transpose(A) * B
        @test @blas(mul!(copy(C), Aclass(spA), transpose(Bclass(spB)))) ≈ A * transpose(B)
        @test @blas(mul!(copy(C), transpose(Aclass(spA)), transpose(Bclass(spB)))) ≈ transpose(A) * transpose(B)

        # adjoint of symmetric/triangular complex matrices seems to be not implemented
        if (T <: Complex) && (
            (Aclass <: LinearAlgebra.AbstractTriangular) || (Aclass <: Symmetric))
            @test_throws MKLSparseError mul!(copy(C), Aclass(spA)', Bclass(spB), α, β)
        else
            @test @blas(mul!(copy(C), Aclass(spA)', Bclass(spB), α, β)) ≈ mul!(copy(C), A', B, α, β)
            @test @blas(mul!(copy(C), Aclass(spA)', Bclass(spB))) ≈ A' * B
        end

        if (T <: Complex) && (
            (Bclass <: LinearAlgebra.AbstractTriangular) || (Bclass <: Symmetric))
            @test_throws MKLSparseError mul!(copy(C), Aclass(spA), Bclass(spB)', α, β)
        else
            @test @blas(mul!(copy(C), Aclass(spA), Bclass(spB)', α, β)) ≈ mul!(copy(C), A, B', α, β)
            @test @blas(mul!(copy(C), Aclass(spA), Bclass(spB)')) ≈ A * B'
        end

        if (T <: Complex) && (
            (Aclass <: LinearAlgebra.AbstractTriangular) || (Aclass <: Symmetric) ||
            (Bclass <: LinearAlgebra.AbstractTriangular) || (Bclass <: Symmetric))
            @test_throws MKLSparseError mul!(copy(C), Aclass(spA)', Bclass(spB)', α, β)
        else
            @test @blas(mul!(copy(C), Aclass(spA)', Bclass(spB)', α, β)) ≈ mul!(copy(C), A', B', α, β)
            @test @blas(mul!(copy(C), Aclass(spA)', Bclass(spB)')) ≈ A' * B'
        end
    end
end

# sparse := sparse * sparse for special matrices
@testset "mul!(sparse, $Aclass{$SPMT{$T}}, $Bclass{$SPMT{$T}})" for
        (Aclass, convert_to_Aclass) in matrix_classes,
        (Bclass, convert_to_Bclass) in matrix_classes

    for _ in 1:ntries
        n = rand(10:50)
        spf = 0.1 + 0.8 * rand()

        spA = Aclass(convert_to_Aclass(sparserandn(SPMT{T, IT}, n, n, spf)))
        spB = Bclass(convert_to_Bclass(sparserandn(SPMT{T, IT}, n, n, spf)))
        A = convert(Matrix, spA)
        B = convert(Matrix, spB)

        # sparse result of the spA*spB
        # cannot convert dense into sparse since it generates sorted rowvals,
        # while MKLSparse generates non-sorted rowvals
        # keep the sparsity pattern, but randomize the nzvalues
        spAB = convert(SPMT{T,IT}, spA * spB)
        nonzeros(spAB) .= randn(T, nnz(spAB))
        sptAB = convert(SPMT{T,IT}, transpose(spA) * spB)
        nonzeros(sptAB) .= randn(T, nnz(sptAB))
        spAtB = convert(SPMT{T,IT}, spA * transpose(spB))
        nonzeros(spAtB) .= randn(T, nnz(spAtB))
        sptAtB = convert(SPMT{T,IT}, transpose(spA) * transpose(spB))
        nonzeros(sptAtB) .= randn(T, nnz(sptAtB))
        spAtA = convert(SPMT{T,IT}, spA * transpose(spA))
        nonzeros(spAtA) .= randn(T, nnz(spAtA))
        sptAA = convert(SPMT{T,IT}, transpose(spA) * spA)
        nonzeros(sptAA) .= randn(T, nnz(sptAA))
        α, β = denserandn(T, 2)

        @test @blas(spA * spB) isa SPMT{T,IT} # test that the result is sparse
        @test convert(Matrix, @blas(spA * spB)) ≈ A * B atol=3*atol
        @test convert(Matrix, @blas(mul!(copy(spAB), spA, spB))) ≈ A * B atol=3*atol

        # 5-arg mul!() is not implemented

        # cannot in-place multiply into the sparse matrix with the different sparsity pattern
        @test_throws ErrorException mul!(sparserandn(SPMT{T, IT}, n, n, spf), spA, spB)

        for trans in (transpose, adjoint)
            @test convert(Matrix, @blas(spA * trans(spB))) ≈ A * trans(B) atol=3*atol
            @test convert(Matrix, @blas(trans(spA) * spB)) ≈ trans(A) * B atol=3*atol
            @test convert(Matrix, @blas(trans(spA) * trans(spB))) ≈ trans(A) * trans(B) atol=3*atol

            @test convert(Matrix, @blas(mul!(copy(sptAB), trans(spA), spB))) ≈ trans(A) * B atol=3*atol
            @test convert(Matrix, @blas(mul!(copy(spAtB), spA, trans(spB)))) ≈ A * trans(B) atol=3*atol
            @test convert(Matrix, @blas(mul!(copy(sptAtB), trans(spA), trans(spB)))) ≈ trans(A) * trans(B) atol=3*atol

            # 5-arg mul!() is not implemented

            # matrix multiplication with itself
            @test convert(Matrix, @blas(mul!(copy(spAtA), spA, trans(spA)))) ≈ A * trans(A) atol=3*atol
            @test convert(Matrix, @blas(mul!(copy(sptAA), trans(spA), spA))) ≈ trans(A) * A atol=3*atol
        end
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

end # SPMT <: MKLSparse.SparseMatrixCSC

end
