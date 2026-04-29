using Test
using LinearAlgebra
using SparseArrays
using Random
using MKLSparse

# Build deterministic CSC inputs for SpGEMM stage tests.
# Use Int32 indices to match MKL's LP64 interface (the upstream test_BLAS.jl pattern).
function _two_mats(::Type{T}; m = 30, k = 40, n = 25, p = 0.10, seed = 0xa5a5) where T
    rng = MersenneTwister(seed)
    A = convert(SparseMatrixCSC{T, Int32}, sprand(rng, T, m, k, p))
    B = convert(SparseMatrixCSC{T, Int32}, sprand(rng, T, k, n, p))
    return A, B
end

@testset "inspector-executor public API" begin

@testset "MKLSparseMatrix handle round-trip ($T)" for T in (Float32, Float64)
    A, _ = _two_mats(T)
    hA = MKLSparseMatrix(A)
    @test hA.handle != C_NULL

    A_back = convert(SparseMatrixCSC, hA)
    @test size(A_back) == size(A)
    @test nnz(A_back) == nnz(A)
    @test Matrix(A_back) ≈ Matrix(A)

    @test destroy(hA) == MKLSparse.SPARSE_STATUS_SUCCESS
end

@testset "optimize! is idempotent ($T)" for T in (Float32, Float64)
    A, _ = _two_mats(T)
    hA = MKLSparseMatrix(A)
    @test optimize!(hA) === hA
    @test optimize!(hA) === hA
    @test optimize!(hA) === hA
    destroy(hA)
end

@testset "hint setters preserve the matrix ($T)" for T in (Float32, Float64)
    A, _ = _two_mats(T)
    descr_general = matrix_descr('G', 'F', 'N')

    hA = MKLSparseMatrix(A)
    @test set_mv_hint!(hA;     operation='N', descr=descr_general, expected_calls=1000) === hA
    @test set_mm_hint!(hA;     operation='N', descr=descr_general, layout='C',
                                              dense_cols=8, expected_calls=100) === hA
    @test set_dotmv_hint!(hA;  operation='N', descr=descr_general, expected_calls=10) === hA
    @test set_memory_hint!(hA; policy=:aggressive) === hA
    optimize!(hA)

    # Round-trip the optimized handle and confirm it still represents A.
    A_back = convert(SparseMatrixCSC, hA)
    @test Matrix(A_back) ≈ Matrix(A)

    # The plain * dispatch path should also still work after we've created (and
    # optimized) a handle for A — they don't share state but this guards against
    # init-time regressions.
    x = rand(T, size(A, 2))
    @test A * x ≈ Matrix(A) * x
    destroy(hA)
end

@testset "set_memory_hint! :none is accepted ($T)" for T in (Float64,)
    A, _ = _two_mats(T)
    hA = MKLSparseMatrix(A)
    @test set_memory_hint!(hA; policy=:none) === hA
    destroy(hA)
end

@testset "two-stage sp2m matches one-shot ($T)" for T in (Float32, Float64)
    A, B = _two_mats(T)
    descr_general = matrix_descr('G', 'F', 'N')
    C_ref = A * B    # reference SpGEMM via standard Julia path

    # One-shot full-multiplication path.
    hA = MKLSparseMatrix(A)
    hB = MKLSparseMatrix(B)
    hC_full = sp2m_full_mult('N', hA, descr_general, 'N', hB, descr_general)
    C_full = convert(SparseMatrixCSC, hC_full)
    destroy(hC_full)
    @test Matrix(C_full) ≈ Matrix(C_ref)

    # Split path: NNZ_COUNT then FINALIZE_MULT.
    hC = sp2m_nnz_count('N', hA, descr_general, 'N', hB, descr_general)
    @test hC.handle != C_NULL
    sp2m_finalize_mult!(hC, 'N', hA, descr_general, 'N', hB, descr_general)
    C_split = convert(SparseMatrixCSC, hC)
    destroy(hC)

    @test Matrix(C_split) ≈ Matrix(C_ref)
    @test nnz(C_split) == nnz(C_full)

    destroy(hA)
    destroy(hB)
end

@testset "stage 2 is repeatable on the same pattern ($T)" for T in (Float64,)
    A, B = _two_mats(T)
    descr_general = matrix_descr('G', 'F', 'N')

    hA = MKLSparseMatrix(A)
    hB = MKLSparseMatrix(B)
    hC = sp2m_nnz_count('N', hA, descr_general, 'N', hB, descr_general)

    sp2m_finalize_mult!(hC, 'N', hA, descr_general, 'N', hB, descr_general)
    C1 = convert(SparseMatrixCSC, hC)

    sp2m_finalize_mult!(hC, 'N', hA, descr_general, 'N', hB, descr_general)
    C2 = convert(SparseMatrixCSC, hC)

    @test Matrix(C1) ≈ Matrix(C2)

    destroy(hC)
    destroy(hA)
    destroy(hB)
end

end # testset
