@enum MKL_LAYOUT::UInt32 begin
    MKL_ROW_MAJOR = 101
    MKL_COL_MAJOR = 102
end

@enum MKL_TRANSPOSE::UInt32 begin
    MKL_NOTRANS = 111
    MKL_TRANS = 112
    MKL_CONJTRANS = 113
    MKL_CONJ = 114
end

@enum MKL_UPLO::UInt32 begin
    MKL_UPPER = 121
    MKL_LOWER = 122
end

@enum MKL_DIAG::UInt32 begin
    MKL_NONUNIT = 131
    MKL_UNIT = 132
end

@enum MKL_SIDE::UInt32 begin
    MKL_LEFT = 141
    MKL_RIGHT = 142
end

@enum MKL_COMPACT_PACK::UInt32 begin
    MKL_COMPACT_SSE = 181
    MKL_COMPACT_AVX = 182
    MKL_COMPACT_AVX512 = 183
end

const sgemm_jit_kernel_t = Ptr{Cvoid}

const dgemm_jit_kernel_t = Ptr{Cvoid}

const cgemm_jit_kernel_t = Ptr{Cvoid}

const zgemm_jit_kernel_t = Ptr{Cvoid}

@enum mkl_jit_status_t::UInt32 begin
    MKL_JIT_SUCCESS = 0
    MKL_NO_JIT = 1
    MKL_JIT_ERROR = 2
end

function mkl_scsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_scsrmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_scsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_scsrsv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_scsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_scsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_scsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_scsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_scsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_scsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, a::Ptr{Float32}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_scsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scscmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_scscmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_scscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_scscsv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_scoomv(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.mkl_scoomv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_scoosv(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_scoosv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_scoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_scoogemv(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_scoogemv(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_scoosymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_scoosymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ref{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_scootrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_scootrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ref{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_sdiamv(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.mkl_sdiamv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sdiasv(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_sdiasv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, lval::Ref{BlasInt},
                                idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_sdiagemv(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_sdiagemv(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sdiasymv(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_sdiasymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sdiatrsv(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_sdiatrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{Float32}, lval::Ref{BlasInt},
                                  idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_sskymv(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.mkl_sskymv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, x::Ptr{Float32}, beta::Ref{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_sskysv(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.mkl_sskysv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, pntr::Ptr{BlasInt},
                                x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrmv(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_sbsrmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float32}, beta::Ref{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_sbsrsv(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_sbsrsv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_sbsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_sbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_sbsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_sbsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_sbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_sbsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_sbsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, lb::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_sbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_sbsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_scsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_scsrmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_scsrsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_scsrsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_scscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_scscmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_scscsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_scscsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_scoomm(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_scoomm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                nnz::Ref{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_scoosm(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_scoosm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_sdiamm(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_sdiamm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_sdiasm(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_sdiasm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_sskysm(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_sskysm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_sskymm(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.mkl_sskymm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, pntr::Ptr{BlasInt}, b::Ptr{Float32},
                                ldb::Ref{BlasInt}, beta::Ref{Float32}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_sbsrmm(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_sbsrmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, lb::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, b::Ptr{Float32},
                                ldb::Ref{BlasInt}, beta::Ref{Float32}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_sbsrsm(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_sbsrsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSRMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_SCSRMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SCSRSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_SCSRSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SCSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SCSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SCSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, a::Ptr{Float32}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCSCMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_SCSCMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SCSCSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_SCSCSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SCOOMV(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.MKL_SCOOMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SCOOSV(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_SCOOSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_SCOOGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCOOGEMV(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_SCOOSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCOOSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ref{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_SCOOTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCOOTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ref{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SDIAMV(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.MKL_SDIAMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SDIASV(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_SDIASV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, lval::Ref{BlasInt},
                                idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SDIAGEMV(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_SDIAGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SDIASYMV(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_SDIASYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SDIATRSV(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_SDIATRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{Float32}, lval::Ref{BlasInt},
                                  idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_SSKYMV(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.MKL_SSKYMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, x::Ptr{Float32}, beta::Ref{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SSKYSV(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.MKL_SSKYSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, pntr::Ptr{BlasInt},
                                x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRMV(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_SBSRMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float32}, beta::Ref{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SBSRSV(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_SBSRSV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SBSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SBSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SBSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SBSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SBSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, lb::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SBSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SCSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SCSRMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSRSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_SCSRSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSCMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SCSCMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSCSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_SCSCSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCOOMM(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SCOOMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                nnz::Ref{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCOOSM(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_SCOOSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SDIAMM(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SDIAMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SDIASM(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_SDIASM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SSKYSM(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_SSKYSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{Cchar}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SSKYMM(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.MKL_SSKYMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, pntr::Ptr{BlasInt}, b::Ptr{Float32},
                                ldb::Ref{BlasInt}, beta::Ref{Float32}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SBSRMM(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SBSRMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, lb::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{Cchar}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, b::Ptr{Float32},
                                ldb::Ref{BlasInt}, beta::Ref{Float32}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SBSRSM(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_SBSRSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{Cchar},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_dcsrmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_dcsrsv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dcsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dcsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dcsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, a::Ptr{Float64}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcscmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_dcscmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_dcscsv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcoomv(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.mkl_dcoomv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                nnz::Ref{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcoosv(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_dcoosv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_dcoogemv(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcoogemv(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dcoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_dcoosymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcoosymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ref{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_dcootrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcootrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ref{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_ddiamv(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.mkl_ddiamv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_ddiasv(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_ddiasv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, lval::Ref{BlasInt},
                                idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_ddiagemv(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_ddiagemv(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_ddiasymv(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_ddiasymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_ddiatrsv(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_ddiatrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{Float64}, lval::Ref{BlasInt},
                                  idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_dskymv(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.mkl_dskymv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, pntr::Ptr{BlasInt}, x::Ptr{Float64},
                                beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dskysv(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.mkl_dskysv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, pntr::Ptr{BlasInt},
                                x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrmv(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_dbsrmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dbsrsv(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_dbsrsv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dbsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dbsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dbsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dbsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dbsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, lb::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dbsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_dcsrmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcsrsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_dcsrsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_dcscmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcscsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_dcscsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcoomm(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_dcoomm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                nnz::Ref{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcoosm(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_dcoosm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                nnz::Ref{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ddiamm(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_ddiamm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ddiasm(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_ddiasm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dskysm(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_dskysm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, pntr::Ptr{BlasInt}, b::Ptr{Float64},
                                ldb::Ref{BlasInt}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dskymm(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.mkl_dskymm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, pntr::Ptr{BlasInt}, b::Ptr{Float64},
                                ldb::Ref{BlasInt}, beta::Ref{Float64}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_dbsrmm(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_dbsrmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, lb::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, b::Ptr{Float64},
                                ldb::Ref{BlasInt}, beta::Ref{Float64}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_dbsrsm(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_dbsrsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSRMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_DCSRMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCSRSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_DCSRSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DCSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DCSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DCSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, a::Ptr{Float64}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCSCMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_DCSCMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCSCSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_DCSCSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCOOMV(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.MKL_DCOOMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                nnz::Ref{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCOOSV(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_DCOOSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_DCOOGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCOOGEMV(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_DCOOSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCOOSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ref{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_DCOOTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCOOTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ref{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DDIAMV(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.MKL_DDIAMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DDIASV(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_DDIASV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, lval::Ref{BlasInt},
                                idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DDIAGEMV(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_DDIAGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DDIASYMV(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_DDIASYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DDIATRSV(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_DDIATRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{Float64}, lval::Ref{BlasInt},
                                  idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_DSKYMV(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.MKL_DSKYMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, pntr::Ptr{BlasInt}, x::Ptr{Float64},
                                beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DSKYSV(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.MKL_DSKYSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, pntr::Ptr{BlasInt},
                                x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRMV(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_DBSRMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DBSRSV(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_DBSRSV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DBSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DBSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DBSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DBSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DBSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, lb::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DBSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DCSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DCSRMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSRSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_DCSRSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSCMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DCSCMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSCSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_DCSCSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCOOMM(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DCOOMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                nnz::Ref{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCOOSM(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_DCOOSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                nnz::Ref{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DDIAMM(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DDIAMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DDIASM(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_DDIASM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DSKYSM(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_DSKYSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, pntr::Ptr{BlasInt}, b::Ptr{Float64},
                                ldb::Ref{BlasInt}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DSKYMM(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.MKL_DSKYMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, pntr::Ptr{BlasInt}, b::Ptr{Float64},
                                ldb::Ref{BlasInt}, beta::Ref{Float64}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_DBSRMM(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DBSRMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, lb::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{Cchar}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, b::Ptr{Float64},
                                ldb::Ref{BlasInt}, beta::Ref{Float64}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_DBSRSM(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_DBSRSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{Cchar},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_ccsrmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_ccsrsv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_ccsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_ccsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_ccsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccscmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_ccscmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_ccscsv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccoomv(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.mkl_ccoomv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccoosv(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_ccoosv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_ccoogemv(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccoogemv(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_ccoosymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccoosymv(uplo::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_ccootrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccootrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiamv(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.mkl_cdiamv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiasv(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_cdiasv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiagemv(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_cdiagemv(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiasymv(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_cdiasymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiatrsv(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_cdiatrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF32}, lval::Ref{BlasInt},
                                  idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cskymv(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.mkl_cskymv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, beta::Ref{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_cskysv(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.mkl_cskysv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                pntr::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrmv(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_cbsrmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, beta::Ref{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrsv(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_cbsrsv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cbsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{ComplexF32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_cbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_cbsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cbsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{ComplexF32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_cbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_cbsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cbsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, lb::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_cbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_cbsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_ccsrmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccsrsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_ccsrsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_ccscmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccscsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_ccscsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccoomm(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_ccoomm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccoosm(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_ccoosm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, b::Ptr{ComplexF32},
                                ldb::Ref{BlasInt}, c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_cdiamm(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_cdiamm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_cdiasm(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_cdiasm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_cskysm(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_cskysm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_cskymm(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.mkl_cskymm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                pntr::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_cbsrmm(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_cbsrmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, lb::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_cbsrsm(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_cbsrsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSRMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_CCSRMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_CCSRSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CCSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CCSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CCSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSCMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_CCSCMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSCSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_CCSCSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOMV(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.MKL_CCOOMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOSV(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CCOOSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CCOOGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCOOGEMV(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CCOOSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCOOSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CCOOTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCOOTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIAMV(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.MKL_CDIAMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIASV(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_CDIASV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIAGEMV(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_CDIAGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIASYMV(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_CDIASYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIATRSV(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_CDIATRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF32}, lval::Ref{BlasInt},
                                  idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSKYMV(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.MKL_CSKYMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, beta::Ref{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSKYSV(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.MKL_CSKYSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                pntr::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRMV(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_CBSRMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, beta::Ref{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRSV(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_CBSRSV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CBSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{ComplexF32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CBSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CBSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{ComplexF32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CBSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CBSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, lb::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CBSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CCSRMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSRSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_CCSRSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSCMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CCSCMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSCSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_CCSCSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCOOMM(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CCOOMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCOOSM(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_CCOOSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, b::Ptr{ComplexF32},
                                ldb::Ref{BlasInt}, c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CDIAMM(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CDIAMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CDIASM(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_CDIASM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CSKYSM(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_CSKYSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CSKYMM(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.MKL_CSKYMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                pntr::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CBSRMM(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CBSRMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, lb::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CBSRSM(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_CBSRSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_zcsrmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_zcsrsv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zcsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zcsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zcsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcscmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_zcscmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_zcscsv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcoomv(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.mkl_zcoomv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcoosv(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_zcoosv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_zcoogemv(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcoogemv(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_zcoosymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcoosymv(uplo::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_zcootrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcootrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiamv(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.mkl_zdiamv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiasv(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_zdiasv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiagemv(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_zdiagemv(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiasymv(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_zdiasymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiatrsv(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_zdiatrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF64}, lval::Ref{BlasInt},
                                  idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_zskymv(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.mkl_zskymv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, beta::Ref{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function mkl_zskysv(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.mkl_zskysv(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                pntr::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrmv(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_zbsrmv(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, beta::Ref{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrsv(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_zbsrsv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zbsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{ComplexF64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zbsrgemv(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zbsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{ComplexF64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zbsrsymv(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zbsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, lb::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zbsrtrsv(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zcsrmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcsrsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_zcsrsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zcscmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcscsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_zcscsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcoomm(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zcoomm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcoosm(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_zcoosm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, b::Ptr{ComplexF64},
                                ldb::Ref{BlasInt}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zdiamm(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zdiamm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zdiasm(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_zdiasm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zskysm(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_zskysm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zskymm(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.mkl_zskymm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                pntr::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zbsrmm(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zbsrmm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, lb::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zbsrsm(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_zbsrsm(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_ZCSRMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_ZCSRSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZCSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZCSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZCSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSCMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_ZCSCMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSCSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_ZCSCSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOMV(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.MKL_ZCOOMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOSV(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_ZCOOSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_ZCOOGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCOOGEMV(transa::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_ZCOOSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCOOSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_ZCOOTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCOOTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIAMV(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.MKL_ZDIAMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIASV(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_ZDIASV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIAGEMV(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_ZDIAGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIASYMV(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_ZDIASYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIATRSV(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_ZDIATRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF64}, lval::Ref{BlasInt},
                                  idiag::Ref{BlasInt}, ndiag::Ref{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZSKYMV(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.MKL_ZSKYMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, beta::Ref{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZSKYSV(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.MKL_ZSKYSV(transa::Ref{Cchar}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                pntr::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRMV(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_ZBSRMV(transa::Ref{Cchar}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, beta::Ref{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRSV(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_ZBSRSV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZBSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{ComplexF64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZBSRGEMV(transa::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZBSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                  a::Ptr{ComplexF64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZBSRSYMV(uplo::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZBSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar}, diag::Ref{Cchar},
                                  m::Ref{BlasInt}, lb::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZBSRTRSV(uplo::Ref{Cchar}, transa::Ref{Cchar},
                                          diag::Ref{Cchar}, m::Ref{BlasInt}, lb::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZCSRMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_ZCSRSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSCMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZCSCMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSCSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_ZCSCSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCOOMM(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZCOOMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCOOSM(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_ZCOOSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ref{BlasInt}, b::Ptr{ComplexF64},
                                ldb::Ref{BlasInt}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZDIAMM(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZDIAMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                lval::Ref{BlasInt}, idiag::Ref{BlasInt}, ndiag::Ref{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZDIASM(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_ZDIASM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, lval::Ref{BlasInt}, idiag::Ref{BlasInt},
                                ndiag::Ref{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZSKYSM(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_ZSKYSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{Cchar},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZSKYMM(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.MKL_ZSKYMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                pntr::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZBSRMM(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZBSRMM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                k::Ref{BlasInt}, lb::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZBSRSM(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_ZBSRSM(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                lb::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{Cchar}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcsrbsr(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.mkl_dcsrbsr(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ref{BlasInt}, Acsr::Ptr{Float64}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{Float64}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_dcsrcoo(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.mkl_dcsrcoo(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                 Acoo::Ptr{Float64}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function mkl_ddnscsr(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.mkl_ddnscsr(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{Float64}, lda::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ::Ptr{BlasInt}, AI::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_dcsrcsc(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.mkl_dcsrcsc(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{Float64},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_dcsrdia(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.mkl_dcsrdia(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{Float64},
                                 ndiag::Ref{BlasInt}, distance::Ptr{BlasInt}, idiag::Ref{BlasInt},
                                 Acsr_rem::Ref{Float64}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_dcsrsky(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.mkl_dcsrsky(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{Float64},
                                 pointers::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_scsrbsr(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.mkl_scsrbsr(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ref{BlasInt}, Acsr::Ptr{Float32}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{Float32}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_scsrcoo(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.mkl_scsrcoo(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                 Acoo::Ptr{Float32}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function mkl_sdnscsr(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.mkl_sdnscsr(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{Float32}, lda::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ::Ptr{BlasInt}, AI::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_scsrcsc(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.mkl_scsrcsc(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{Float32},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_scsrdia(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.mkl_scsrdia(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{Float32},
                                 ndiag::Ref{BlasInt}, distance::Ptr{BlasInt}, idiag::Ref{BlasInt},
                                 Acsr_rem::Ref{Float32}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_scsrsky(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.mkl_scsrsky(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{Float32},
                                 pointers::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_ccsrbsr(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.mkl_ccsrbsr(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ref{BlasInt}, Acsr::Ptr{ComplexF32}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{ComplexF32}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_ccsrcoo(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.mkl_ccsrcoo(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                 Acoo::Ptr{ComplexF32}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function mkl_cdnscsr(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.mkl_cdnscsr(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{ComplexF32}, lda::Ref{BlasInt},
                                 Acsr::Ptr{ComplexF32}, AJ::Ptr{BlasInt}, AI::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function mkl_ccsrcsc(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.mkl_ccsrcsc(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{ComplexF32},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_ccsrdia(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.mkl_ccsrdia(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{ComplexF32},
                                 ndiag::Ref{BlasInt}, distance::Ptr{BlasInt}, idiag::Ref{BlasInt},
                                 Acsr_rem::Ref{ComplexF32}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_ccsrsky(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.mkl_ccsrsky(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{ComplexF32},
                                 pointers::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_zcsrbsr(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.mkl_zcsrbsr(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ref{BlasInt}, Acsr::Ptr{ComplexF64}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{ComplexF64}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_zcsrcoo(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.mkl_zcsrcoo(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                 Acoo::Ptr{ComplexF64}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function mkl_zdnscsr(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.mkl_zdnscsr(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{ComplexF64}, lda::Ref{BlasInt},
                                 Acsr::Ptr{ComplexF64}, AJ::Ptr{BlasInt}, AI::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function mkl_zcsrcsc(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.mkl_zcsrcsc(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{ComplexF64},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_zcsrdia(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.mkl_zcsrdia(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{ComplexF64},
                                 ndiag::Ref{BlasInt}, distance::Ptr{BlasInt}, idiag::Ref{BlasInt},
                                 Acsr_rem::Ref{ComplexF64}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_zcsrsky(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.mkl_zcsrsky(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{ComplexF64},
                                 pointers::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_DCSRBSR(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.MKL_DCSRBSR(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ref{BlasInt}, Acsr::Ptr{Float64}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{Float64}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_DCSRCOO(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.MKL_DCSRCOO(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                 Acoo::Ptr{Float64}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function MKL_DDNSCSR(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.MKL_DDNSCSR(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{Float64}, lda::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ::Ptr{BlasInt}, AI::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_DCSRCSC(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.MKL_DCSRCSC(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{Float64},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_DCSRDIA(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.MKL_DCSRDIA(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{Float64},
                                 ndiag::Ref{BlasInt}, distance::Ptr{BlasInt}, idiag::Ref{BlasInt},
                                 Acsr_rem::Ref{Float64}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_DCSRSKY(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.MKL_DCSRSKY(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{Float64},
                                 pointers::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_SCSRBSR(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.MKL_SCSRBSR(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ref{BlasInt}, Acsr::Ptr{Float32}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{Float32}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_SCSRCOO(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.MKL_SCSRCOO(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                 Acoo::Ptr{Float32}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function MKL_SDNSCSR(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.MKL_SDNSCSR(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{Float32}, lda::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ::Ptr{BlasInt}, AI::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_SCSRCSC(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.MKL_SCSRCSC(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{Float32},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_SCSRDIA(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.MKL_SCSRDIA(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{Float32},
                                 ndiag::Ref{BlasInt}, distance::Ptr{BlasInt}, idiag::Ref{BlasInt},
                                 Acsr_rem::Ref{Float32}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_SCSRSKY(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.MKL_SCSRSKY(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{Float32},
                                 pointers::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_CCSRBSR(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.MKL_CCSRBSR(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ref{BlasInt}, Acsr::Ptr{ComplexF32}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{ComplexF32}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_CCSRCOO(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.MKL_CCSRCOO(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                 Acoo::Ptr{ComplexF32}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function MKL_CDNSCSR(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.MKL_CDNSCSR(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{ComplexF32}, lda::Ref{BlasInt},
                                 Acsr::Ptr{ComplexF32}, AJ::Ptr{BlasInt}, AI::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function MKL_CCSRCSC(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.MKL_CCSRCSC(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{ComplexF32},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_CCSRDIA(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.MKL_CCSRDIA(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{ComplexF32},
                                 ndiag::Ref{BlasInt}, distance::Ptr{BlasInt}, idiag::Ref{BlasInt},
                                 Acsr_rem::Ref{ComplexF32}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_CCSRSKY(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.MKL_CCSRSKY(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{ComplexF32},
                                 pointers::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRBSR(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.MKL_ZCSRBSR(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ref{BlasInt}, Acsr::Ptr{ComplexF64}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{ComplexF64}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRCOO(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.MKL_ZCSRCOO(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ref{BlasInt},
                                 Acoo::Ptr{ComplexF64}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function MKL_ZDNSCSR(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.MKL_ZDNSCSR(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{ComplexF64}, lda::Ref{BlasInt},
                                 Acsr::Ptr{ComplexF64}, AJ::Ptr{BlasInt}, AI::Ptr{BlasInt},
                                 info::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRCSC(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.MKL_ZCSRCSC(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{ComplexF64},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRDIA(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.MKL_ZCSRDIA(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{ComplexF64},
                                 ndiag::Ref{BlasInt}, distance::Ptr{BlasInt}, idiag::Ref{BlasInt},
                                 Acsr_rem::Ref{ComplexF64}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRSKY(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.MKL_ZCSRSKY(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{ComplexF64},
                                 pointers::Ptr{BlasInt}, info::Ref{BlasInt})::Cvoid
end

function mkl_dcsrmultcsr(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.mkl_dcsrmultcsr(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{Float64}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{Float64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{Float64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function mkl_dcsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.mkl_dcsrmultd(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{Float64}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{Float64}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcsradd(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.mkl_dcsradd(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{Float64}, ja::Ptr{BlasInt},
                                 ia::Ptr{BlasInt}, beta::Ref{Float64}, b::Ptr{Float64},
                                 jb::Ptr{BlasInt}, ib::Ptr{BlasInt}, c::Ptr{Float64},
                                 jc::Ptr{BlasInt}, ic::Ptr{BlasInt}, nnzmax::Ref{BlasInt},
                                 ierr::Ref{BlasInt})::Cvoid
end

function mkl_scsrmultcsr(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.mkl_scsrmultcsr(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{Float32}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{Float32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{Float32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function mkl_scsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.mkl_scsrmultd(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{Float32}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{Float32}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_scsradd(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.mkl_scsradd(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{Float32}, ja::Ptr{BlasInt},
                                 ia::Ptr{BlasInt}, beta::Ref{Float32}, b::Ptr{Float32},
                                 jb::Ptr{BlasInt}, ib::Ptr{BlasInt}, c::Ptr{Float32},
                                 jc::Ptr{BlasInt}, ic::Ptr{BlasInt}, nnzmax::Ref{BlasInt},
                                 ierr::Ref{BlasInt})::Cvoid
end

function mkl_ccsrmultcsr(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.mkl_ccsrmultcsr(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{ComplexF32}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{ComplexF32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{ComplexF32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function mkl_ccsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.mkl_ccsrmultd(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{ComplexF32}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{ComplexF32}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{ComplexF32},
                                   ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccsradd(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.mkl_ccsradd(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{ComplexF32},
                                 ja::Ptr{BlasInt}, ia::Ptr{BlasInt}, beta::Ref{ComplexF32},
                                 b::Ptr{ComplexF32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                 c::Ptr{ComplexF32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                 nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function mkl_zcsrmultcsr(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.mkl_zcsrmultcsr(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{ComplexF64}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{ComplexF64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{ComplexF64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function mkl_zcsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.mkl_zcsrmultd(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{ComplexF64}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{ComplexF64}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{ComplexF64},
                                   ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcsradd(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.mkl_zcsradd(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{ComplexF64},
                                 ja::Ptr{BlasInt}, ia::Ptr{BlasInt}, beta::Ref{ComplexF64},
                                 b::Ptr{ComplexF64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                 c::Ptr{ComplexF64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                 nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function MKL_DCSRMULTCSR(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.MKL_DCSRMULTCSR(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{Float64}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{Float64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{Float64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function MKL_DCSRMULTD(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.MKL_DCSRMULTD(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{Float64}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{Float64}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSRADD(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.MKL_DCSRADD(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{Float64}, ja::Ptr{BlasInt},
                                 ia::Ptr{BlasInt}, beta::Ref{Float64}, b::Ptr{Float64},
                                 jb::Ptr{BlasInt}, ib::Ptr{BlasInt}, c::Ptr{Float64},
                                 jc::Ptr{BlasInt}, ic::Ptr{BlasInt}, nnzmax::Ref{BlasInt},
                                 ierr::Ref{BlasInt})::Cvoid
end

function MKL_SCSRMULTCSR(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.MKL_SCSRMULTCSR(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{Float32}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{Float32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{Float32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function MKL_SCSRMULTD(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.MKL_SCSRMULTD(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{Float32}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{Float32}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSRADD(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.MKL_SCSRADD(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{Float32}, ja::Ptr{BlasInt},
                                 ia::Ptr{BlasInt}, beta::Ref{Float32}, b::Ptr{Float32},
                                 jb::Ptr{BlasInt}, ib::Ptr{BlasInt}, c::Ptr{Float32},
                                 jc::Ptr{BlasInt}, ic::Ptr{BlasInt}, nnzmax::Ref{BlasInt},
                                 ierr::Ref{BlasInt})::Cvoid
end

function MKL_CCSRMULTCSR(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.MKL_CCSRMULTCSR(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{ComplexF32}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{ComplexF32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{ComplexF32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function MKL_CCSRMULTD(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.MKL_CCSRMULTD(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{ComplexF32}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{ComplexF32}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{ComplexF32},
                                   ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSRADD(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.MKL_CCSRADD(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{ComplexF32},
                                 ja::Ptr{BlasInt}, ia::Ptr{BlasInt}, beta::Ref{ComplexF32},
                                 b::Ptr{ComplexF32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                 c::Ptr{ComplexF32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                 nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRMULTCSR(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.MKL_ZCSRMULTCSR(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{ComplexF64}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{ComplexF64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{ComplexF64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRMULTD(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.MKL_ZCSRMULTD(transa::Ref{Cchar}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{ComplexF64}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{ComplexF64}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{ComplexF64},
                                   ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRADD(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.MKL_ZCSRADD(transa::Ref{Cchar}, job::Ref{BlasInt}, sort::Ref{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{ComplexF64},
                                 ja::Ptr{BlasInt}, ia::Ptr{BlasInt}, beta::Ref{ComplexF64},
                                 b::Ptr{ComplexF64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                 c::Ptr{ComplexF64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                 nnzmax::Ref{BlasInt}, ierr::Ref{BlasInt})::Cvoid
end

@enum sparse_status_t::UInt32 begin
    SPARSE_STATUS_SUCCESS = 0
    SPARSE_STATUS_NOT_INITIALIZED = 1
    SPARSE_STATUS_ALLOC_FAILED = 2
    SPARSE_STATUS_INVALID_VALUE = 3
    SPARSE_STATUS_EXECUTION_FAILED = 4
    SPARSE_STATUS_INTERNAL_ERROR = 5
    SPARSE_STATUS_NOT_SUPPORTED = 6
end

@enum sparse_operation_t::UInt32 begin
    SPARSE_OPERATION_NON_TRANSPOSE = 10
    SPARSE_OPERATION_TRANSPOSE = 11
    SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12
end

@enum sparse_matrix_type_t::UInt32 begin
    SPARSE_MATRIX_TYPE_GENERAL = 20
    SPARSE_MATRIX_TYPE_SYMMETRIC = 21
    SPARSE_MATRIX_TYPE_HERMITIAN = 22
    SPARSE_MATRIX_TYPE_TRIANGULAR = 23
    SPARSE_MATRIX_TYPE_DIAGONAL = 24
    SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 25
    SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 26
end

@enum sparse_index_base_t::UInt32 begin
    SPARSE_INDEX_BASE_ZERO = 0
    SPARSE_INDEX_BASE_ONE = 1
end

@enum sparse_fill_mode_t::UInt32 begin
    SPARSE_FILL_MODE_LOWER = 40
    SPARSE_FILL_MODE_UPPER = 41
    SPARSE_FILL_MODE_FULL = 42
end

@enum sparse_diag_type_t::UInt32 begin
    SPARSE_DIAG_NON_UNIT = 50
    SPARSE_DIAG_UNIT = 51
end

@enum sparse_layout_t::UInt32 begin
    SPARSE_LAYOUT_ROW_MAJOR = 101
    SPARSE_LAYOUT_COLUMN_MAJOR = 102
end

@enum verbose_mode_t::UInt32 begin
    SPARSE_VERBOSE_OFF = 70
    SPARSE_VERBOSE_BASIC = 71
    SPARSE_VERBOSE_EXTENDED = 72
end

@enum sparse_memory_usage_t::UInt32 begin
    SPARSE_MEMORY_NONE = 80
    SPARSE_MEMORY_AGGRESSIVE = 81
end

@enum sparse_request_t::UInt32 begin
    SPARSE_STAGE_FULL_MULT = 90
    SPARSE_STAGE_NNZ_COUNT = 91
    SPARSE_STAGE_FINALIZE_MULT = 92
    SPARSE_STAGE_FULL_MULT_NO_VAL = 93
    SPARSE_STAGE_FINALIZE_MULT_NO_VAL = 94
end

@enum sparse_sor_type_t::UInt32 begin
    SPARSE_SOR_FORWARD = 110
    SPARSE_SOR_BACKWARD = 111
    SPARSE_SOR_SYMMETRIC = 112
end

mutable struct sparse_matrix end

const sparse_matrix_t = Ptr{sparse_matrix}

struct matrix_descr
    type::sparse_matrix_type_t
    mode::sparse_fill_mode_t
    diag::sparse_diag_type_t
end

function mkl_sparse_s_create_coo(A, indexing, rows, cols, nnz, row_indx, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_s_create_coo(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, nnz::BlasInt, row_indx::Ptr{BlasInt},
                                             col_indx::Ptr{BlasInt},
                                             values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_create_coo(A, indexing, rows, cols, nnz, row_indx, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_d_create_coo(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, nnz::BlasInt, row_indx::Ptr{BlasInt},
                                             col_indx::Ptr{BlasInt},
                                             values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_create_coo(A, indexing, rows, cols, nnz, row_indx, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_c_create_coo(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, nnz::BlasInt, row_indx::Ptr{BlasInt},
                                             col_indx::Ptr{BlasInt},
                                             values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_create_coo(A, indexing, rows, cols, nnz, row_indx, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_z_create_coo(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, nnz::BlasInt, row_indx::Ptr{BlasInt},
                                             col_indx::Ptr{BlasInt},
                                             values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_create_coo_64(A, indexing, rows, cols, nnz, row_indx, col_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_create_coo_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                nnz::Clonglong, row_indx::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_create_coo_64(A, indexing, rows, cols, nnz, row_indx, col_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_create_coo_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                nnz::Clonglong, row_indx::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_create_coo_64(A, indexing, rows, cols, nnz, row_indx, col_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_create_coo_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                nnz::Clonglong, row_indx::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_create_coo_64(A, indexing, rows, cols, nnz, row_indx, col_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_create_coo_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                nnz::Clonglong, row_indx::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx,
                                 values)
    @ccall libmkl_rt.mkl_sparse_s_create_csr(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, rows_start::Ptr{BlasInt},
                                             rows_end::Ptr{BlasInt}, col_indx::Ptr{BlasInt},
                                             values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx,
                                 values)
    @ccall libmkl_rt.mkl_sparse_d_create_csr(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, rows_start::Ptr{BlasInt},
                                             rows_end::Ptr{BlasInt}, col_indx::Ptr{BlasInt},
                                             values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx,
                                 values)
    @ccall libmkl_rt.mkl_sparse_c_create_csr(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, rows_start::Ptr{BlasInt},
                                             rows_end::Ptr{BlasInt}, col_indx::Ptr{BlasInt},
                                             values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx,
                                 values)
    @ccall libmkl_rt.mkl_sparse_z_create_csr(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, rows_start::Ptr{BlasInt},
                                             rows_end::Ptr{BlasInt}, col_indx::Ptr{BlasInt},
                                             values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_create_csr_64(A, indexing, rows, cols, rows_start, rows_end, col_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_create_csr_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                rows_start::Ptr{Clonglong},
                                                rows_end::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_create_csr_64(A, indexing, rows, cols, rows_start, rows_end, col_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_create_csr_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                rows_start::Ptr{Clonglong},
                                                rows_end::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_create_csr_64(A, indexing, rows, cols, rows_start, rows_end, col_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_create_csr_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                rows_start::Ptr{Clonglong},
                                                rows_end::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_create_csr_64(A, indexing, rows, cols, rows_start, rows_end, col_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_create_csr_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                rows_start::Ptr{Clonglong},
                                                rows_end::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_create_csc(A, indexing, rows, cols, cols_start, cols_end, row_indx,
                                 values)
    @ccall libmkl_rt.mkl_sparse_s_create_csc(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, cols_start::Ptr{BlasInt},
                                             cols_end::Ptr{BlasInt}, row_indx::Ptr{BlasInt},
                                             values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_create_csc(A, indexing, rows, cols, cols_start, cols_end, row_indx,
                                 values)
    @ccall libmkl_rt.mkl_sparse_d_create_csc(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, cols_start::Ptr{BlasInt},
                                             cols_end::Ptr{BlasInt}, row_indx::Ptr{BlasInt},
                                             values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_create_csc(A, indexing, rows, cols, cols_start, cols_end, row_indx,
                                 values)
    @ccall libmkl_rt.mkl_sparse_c_create_csc(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, cols_start::Ptr{BlasInt},
                                             cols_end::Ptr{BlasInt}, row_indx::Ptr{BlasInt},
                                             values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_create_csc(A, indexing, rows, cols, cols_start, cols_end, row_indx,
                                 values)
    @ccall libmkl_rt.mkl_sparse_z_create_csc(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t, rows::BlasInt,
                                             cols::BlasInt, cols_start::Ptr{BlasInt},
                                             cols_end::Ptr{BlasInt}, row_indx::Ptr{BlasInt},
                                             values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_create_csc_64(A, indexing, rows, cols, cols_start, cols_end, row_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_create_csc_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                cols_start::Ptr{Clonglong},
                                                cols_end::Ptr{Clonglong},
                                                row_indx::Ptr{Clonglong},
                                                values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_create_csc_64(A, indexing, rows, cols, cols_start, cols_end, row_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_create_csc_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                cols_start::Ptr{Clonglong},
                                                cols_end::Ptr{Clonglong},
                                                row_indx::Ptr{Clonglong},
                                                values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_create_csc_64(A, indexing, rows, cols, cols_start, cols_end, row_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_create_csc_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                cols_start::Ptr{Clonglong},
                                                cols_end::Ptr{Clonglong},
                                                row_indx::Ptr{Clonglong},
                                                values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_create_csc_64(A, indexing, rows, cols, cols_start, cols_end, row_indx,
                                    values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_create_csc_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                rows::Clonglong, cols::Clonglong,
                                                cols_start::Ptr{Clonglong},
                                                cols_end::Ptr{Clonglong},
                                                row_indx::Ptr{Clonglong},
                                                values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_create_bsr(A, indexing, block_layout, rows, cols, block_size,
                                 rows_start, rows_end, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_s_create_bsr(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t,
                                             block_layout::sparse_layout_t, rows::BlasInt,
                                             cols::BlasInt, block_size::BlasInt,
                                             rows_start::Ptr{BlasInt}, rows_end::Ptr{BlasInt},
                                             col_indx::Ptr{BlasInt},
                                             values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_create_bsr(A, indexing, block_layout, rows, cols, block_size,
                                 rows_start, rows_end, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_d_create_bsr(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t,
                                             block_layout::sparse_layout_t, rows::BlasInt,
                                             cols::BlasInt, block_size::BlasInt,
                                             rows_start::Ptr{BlasInt}, rows_end::Ptr{BlasInt},
                                             col_indx::Ptr{BlasInt},
                                             values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_create_bsr(A, indexing, block_layout, rows, cols, block_size,
                                 rows_start, rows_end, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_c_create_bsr(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t,
                                             block_layout::sparse_layout_t, rows::BlasInt,
                                             cols::BlasInt, block_size::BlasInt,
                                             rows_start::Ptr{BlasInt}, rows_end::Ptr{BlasInt},
                                             col_indx::Ptr{BlasInt},
                                             values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_create_bsr(A, indexing, block_layout, rows, cols, block_size,
                                 rows_start, rows_end, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_z_create_bsr(A::Ptr{sparse_matrix_t},
                                             indexing::sparse_index_base_t,
                                             block_layout::sparse_layout_t, rows::BlasInt,
                                             cols::BlasInt, block_size::BlasInt,
                                             rows_start::Ptr{BlasInt}, rows_end::Ptr{BlasInt},
                                             col_indx::Ptr{BlasInt},
                                             values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_create_bsr_64(A, indexing, block_layout, rows, cols, block_size,
                                    rows_start, rows_end, col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_create_bsr_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                block_layout::sparse_layout_t,
                                                rows::Clonglong, cols::Clonglong,
                                                block_size::Clonglong,
                                                rows_start::Ptr{Clonglong},
                                                rows_end::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_create_bsr_64(A, indexing, block_layout, rows, cols, block_size,
                                    rows_start, rows_end, col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_create_bsr_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                block_layout::sparse_layout_t,
                                                rows::Clonglong, cols::Clonglong,
                                                block_size::Clonglong,
                                                rows_start::Ptr{Clonglong},
                                                rows_end::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_create_bsr_64(A, indexing, block_layout, rows, cols, block_size,
                                    rows_start, rows_end, col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_create_bsr_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                block_layout::sparse_layout_t,
                                                rows::Clonglong, cols::Clonglong,
                                                block_size::Clonglong,
                                                rows_start::Ptr{Clonglong},
                                                rows_end::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_create_bsr_64(A, indexing, block_layout, rows, cols, block_size,
                                    rows_start, rows_end, col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_create_bsr_64(A::Ptr{sparse_matrix_t},
                                                indexing::sparse_index_base_t,
                                                block_layout::sparse_layout_t,
                                                rows::Clonglong, cols::Clonglong,
                                                block_size::Clonglong,
                                                rows_start::Ptr{Clonglong},
                                                rows_end::Ptr{Clonglong},
                                                col_indx::Ptr{Clonglong},
                                                values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_copy(source, descr, dest)
    @ccall libmkl_rt.mkl_sparse_copy(source::sparse_matrix_t, descr::matrix_descr,
                                     dest::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_copy_64(source, descr, dest)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_copy_64(source::sparse_matrix_t, descr::matrix_descr,
                                        dest::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_destroy(A)
    @ccall libmkl_rt.mkl_sparse_destroy(A::sparse_matrix_t)::sparse_status_t
end

function mkl_sparse_destroy_64(A)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_destroy_64(A::sparse_matrix_t)::sparse_status_t
end

function mkl_sparse_get_error_info(A, info)
    @ccall libmkl_rt.mkl_sparse_get_error_info(A::sparse_matrix_t,
                                               info::Ref{BlasInt})::sparse_status_t
end

function mkl_sparse_get_error_info_64(A, info)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_get_error_info_64(A::sparse_matrix_t,
                                                  info::Ptr{Clonglong})::sparse_status_t
end

function mkl_sparse_convert_csr(source, operation, dest)
    @ccall libmkl_rt.mkl_sparse_convert_csr(source::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            dest::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_convert_bsr(source, block_size, block_layout, operation, dest)
    @ccall libmkl_rt.mkl_sparse_convert_bsr(source::sparse_matrix_t, block_size::BlasInt,
                                            block_layout::sparse_layout_t,
                                            operation::sparse_operation_t,
                                            dest::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_convert_csr_64(source, operation, dest)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_convert_csr_64(source::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               dest::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_convert_bsr_64(source, block_size, block_layout, operation, dest)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_convert_bsr_64(source::sparse_matrix_t,
                                               block_size::Clonglong,
                                               block_layout::sparse_layout_t,
                                               operation::sparse_operation_t,
                                               dest::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_s_export_bsr(source, indexing, block_layout, rows, cols, block_size,
                                 rows_start, rows_end, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_s_export_bsr(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             block_layout::Ptr{sparse_layout_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             block_size::Ptr{BlasInt},
                                             rows_start::Ptr{Ptr{BlasInt}},
                                             rows_end::Ptr{Ptr{BlasInt}},
                                             col_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{Float32}})::sparse_status_t
end

function mkl_sparse_d_export_bsr(source, indexing, block_layout, rows, cols, block_size,
                                 rows_start, rows_end, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_d_export_bsr(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             block_layout::Ptr{sparse_layout_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             block_size::Ptr{BlasInt},
                                             rows_start::Ptr{Ptr{BlasInt}},
                                             rows_end::Ptr{Ptr{BlasInt}},
                                             col_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{Float64}})::sparse_status_t
end

function mkl_sparse_c_export_bsr(source, indexing, block_layout, rows, cols, block_size,
                                 rows_start, rows_end, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_c_export_bsr(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             block_layout::Ptr{sparse_layout_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             block_size::Ptr{BlasInt},
                                             rows_start::Ptr{Ptr{BlasInt}},
                                             rows_end::Ptr{Ptr{BlasInt}},
                                             col_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{ComplexF32}})::sparse_status_t
end

function mkl_sparse_z_export_bsr(source, indexing, block_layout, rows, cols, block_size,
                                 rows_start, rows_end, col_indx, values)
    @ccall libmkl_rt.mkl_sparse_z_export_bsr(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             block_layout::Ptr{sparse_layout_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             block_size::Ptr{BlasInt},
                                             rows_start::Ptr{Ptr{BlasInt}},
                                             rows_end::Ptr{Ptr{BlasInt}},
                                             col_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{ComplexF64}})::sparse_status_t
end

function mkl_sparse_s_export_bsr_64(source, indexing, block_layout, rows, cols, block_size,
                                    rows_start, rows_end, col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_export_bsr_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                block_layout::Ptr{sparse_layout_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                block_size::Ptr{Clonglong},
                                                rows_start::Ptr{Ptr{Clonglong}},
                                                rows_end::Ptr{Ptr{Clonglong}},
                                                col_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{Float32}})::sparse_status_t
end

function mkl_sparse_d_export_bsr_64(source, indexing, block_layout, rows, cols, block_size,
                                    rows_start, rows_end, col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_export_bsr_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                block_layout::Ptr{sparse_layout_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                block_size::Ptr{Clonglong},
                                                rows_start::Ptr{Ptr{Clonglong}},
                                                rows_end::Ptr{Ptr{Clonglong}},
                                                col_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{Float64}})::sparse_status_t
end

function mkl_sparse_c_export_bsr_64(source, indexing, block_layout, rows, cols, block_size,
                                    rows_start, rows_end, col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_export_bsr_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                block_layout::Ptr{sparse_layout_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                block_size::Ptr{Clonglong},
                                                rows_start::Ptr{Ptr{Clonglong}},
                                                rows_end::Ptr{Ptr{Clonglong}},
                                                col_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{ComplexF32}})::sparse_status_t
end

function mkl_sparse_z_export_bsr_64(source, indexing, block_layout, rows, cols, block_size,
                                    rows_start, rows_end, col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_export_bsr_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                block_layout::Ptr{sparse_layout_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                block_size::Ptr{Clonglong},
                                                rows_start::Ptr{Ptr{Clonglong}},
                                                rows_end::Ptr{Ptr{Clonglong}},
                                                col_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{ComplexF64}})::sparse_status_t
end

function mkl_sparse_s_export_csr(source, indexing, rows, cols, rows_start, rows_end,
                                 col_indx, values)
    @ccall libmkl_rt.mkl_sparse_s_export_csr(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             rows_start::Ptr{Ptr{BlasInt}},
                                             rows_end::Ptr{Ptr{BlasInt}},
                                             col_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{Float32}})::sparse_status_t
end

function mkl_sparse_d_export_csr(source, indexing, rows, cols, rows_start, rows_end,
                                 col_indx, values)
    @ccall libmkl_rt.mkl_sparse_d_export_csr(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             rows_start::Ptr{Ptr{BlasInt}},
                                             rows_end::Ptr{Ptr{BlasInt}},
                                             col_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{Float64}})::sparse_status_t
end

function mkl_sparse_c_export_csr(source, indexing, rows, cols, rows_start, rows_end,
                                 col_indx, values)
    @ccall libmkl_rt.mkl_sparse_c_export_csr(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             rows_start::Ptr{Ptr{BlasInt}},
                                             rows_end::Ptr{Ptr{BlasInt}},
                                             col_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{ComplexF32}})::sparse_status_t
end

function mkl_sparse_z_export_csr(source, indexing, rows, cols, rows_start, rows_end,
                                 col_indx, values)
    @ccall libmkl_rt.mkl_sparse_z_export_csr(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             rows_start::Ptr{Ptr{BlasInt}},
                                             rows_end::Ptr{Ptr{BlasInt}},
                                             col_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{ComplexF64}})::sparse_status_t
end

function mkl_sparse_s_export_csr_64(source, indexing, rows, cols, rows_start, rows_end,
                                    col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_export_csr_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                rows_start::Ptr{Ptr{Clonglong}},
                                                rows_end::Ptr{Ptr{Clonglong}},
                                                col_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{Float32}})::sparse_status_t
end

function mkl_sparse_d_export_csr_64(source, indexing, rows, cols, rows_start, rows_end,
                                    col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_export_csr_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                rows_start::Ptr{Ptr{Clonglong}},
                                                rows_end::Ptr{Ptr{Clonglong}},
                                                col_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{Float64}})::sparse_status_t
end

function mkl_sparse_c_export_csr_64(source, indexing, rows, cols, rows_start, rows_end,
                                    col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_export_csr_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                rows_start::Ptr{Ptr{Clonglong}},
                                                rows_end::Ptr{Ptr{Clonglong}},
                                                col_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{ComplexF32}})::sparse_status_t
end

function mkl_sparse_z_export_csr_64(source, indexing, rows, cols, rows_start, rows_end,
                                    col_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_export_csr_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                rows_start::Ptr{Ptr{Clonglong}},
                                                rows_end::Ptr{Ptr{Clonglong}},
                                                col_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{ComplexF64}})::sparse_status_t
end

function mkl_sparse_s_export_csc(source, indexing, rows, cols, cols_start, cols_end,
                                 row_indx, values)
    @ccall libmkl_rt.mkl_sparse_s_export_csc(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             cols_start::Ptr{Ptr{BlasInt}},
                                             cols_end::Ptr{Ptr{BlasInt}},
                                             row_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{Float32}})::sparse_status_t
end

function mkl_sparse_d_export_csc(source, indexing, rows, cols, cols_start, cols_end,
                                 row_indx, values)
    @ccall libmkl_rt.mkl_sparse_d_export_csc(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             cols_start::Ptr{Ptr{BlasInt}},
                                             cols_end::Ptr{Ptr{BlasInt}},
                                             row_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{Float64}})::sparse_status_t
end

function mkl_sparse_c_export_csc(source, indexing, rows, cols, cols_start, cols_end,
                                 row_indx, values)
    @ccall libmkl_rt.mkl_sparse_c_export_csc(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             cols_start::Ptr{Ptr{BlasInt}},
                                             cols_end::Ptr{Ptr{BlasInt}},
                                             row_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{ComplexF32}})::sparse_status_t
end

function mkl_sparse_z_export_csc(source, indexing, rows, cols, cols_start, cols_end,
                                 row_indx, values)
    @ccall libmkl_rt.mkl_sparse_z_export_csc(source::sparse_matrix_t,
                                             indexing::Ptr{sparse_index_base_t},
                                             rows::Ptr{BlasInt}, cols::Ptr{BlasInt},
                                             cols_start::Ptr{Ptr{BlasInt}},
                                             cols_end::Ptr{Ptr{BlasInt}},
                                             row_indx::Ptr{Ptr{BlasInt}},
                                             values::Ptr{Ptr{ComplexF64}})::sparse_status_t
end

function mkl_sparse_s_export_csc_64(source, indexing, rows, cols, cols_start, cols_end,
                                    row_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_export_csc_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                cols_start::Ptr{Ptr{Clonglong}},
                                                cols_end::Ptr{Ptr{Clonglong}},
                                                row_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{Float32}})::sparse_status_t
end

function mkl_sparse_d_export_csc_64(source, indexing, rows, cols, cols_start, cols_end,
                                    row_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_export_csc_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                cols_start::Ptr{Ptr{Clonglong}},
                                                cols_end::Ptr{Ptr{Clonglong}},
                                                row_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{Float64}})::sparse_status_t
end

function mkl_sparse_c_export_csc_64(source, indexing, rows, cols, cols_start, cols_end,
                                    row_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_export_csc_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                cols_start::Ptr{Ptr{Clonglong}},
                                                cols_end::Ptr{Ptr{Clonglong}},
                                                row_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{ComplexF32}})::sparse_status_t
end

function mkl_sparse_z_export_csc_64(source, indexing, rows, cols, cols_start, cols_end,
                                    row_indx, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_export_csc_64(source::sparse_matrix_t,
                                                indexing::Ptr{sparse_index_base_t},
                                                rows::Ptr{Clonglong}, cols::Ptr{Clonglong},
                                                cols_start::Ptr{Ptr{Clonglong}},
                                                cols_end::Ptr{Ptr{Clonglong}},
                                                row_indx::Ptr{Ptr{Clonglong}},
                                                values::Ptr{Ptr{ComplexF64}})::sparse_status_t
end

function mkl_sparse_s_set_value(A, row, col, value)
    @ccall libmkl_rt.mkl_sparse_s_set_value(A::sparse_matrix_t, row::BlasInt, col::BlasInt,
                                            value::Float32)::sparse_status_t
end

function mkl_sparse_d_set_value(A, row, col, value)
    @ccall libmkl_rt.mkl_sparse_d_set_value(A::sparse_matrix_t, row::BlasInt, col::BlasInt,
                                            value::Float64)::sparse_status_t
end

function mkl_sparse_c_set_value(A, row, col, value)
    @ccall libmkl_rt.mkl_sparse_c_set_value(A::sparse_matrix_t, row::BlasInt, col::BlasInt,
                                            value::ComplexF32)::sparse_status_t
end

function mkl_sparse_z_set_value(A, row, col, value)
    @ccall libmkl_rt.mkl_sparse_z_set_value(A::sparse_matrix_t, row::BlasInt, col::BlasInt,
                                            value::ComplexF64)::sparse_status_t
end

function mkl_sparse_s_set_value_64(A, row, col, value)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_set_value_64(A::sparse_matrix_t, row::Clonglong,
                                               col::Clonglong,
                                               value::Float32)::sparse_status_t
end

function mkl_sparse_d_set_value_64(A, row, col, value)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_set_value_64(A::sparse_matrix_t, row::Clonglong,
                                               col::Clonglong,
                                               value::Float64)::sparse_status_t
end

function mkl_sparse_c_set_value_64(A, row, col, value)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_set_value_64(A::sparse_matrix_t, row::Clonglong,
                                               col::Clonglong,
                                               value::ComplexF32)::sparse_status_t
end

function mkl_sparse_z_set_value_64(A, row, col, value)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_set_value_64(A::sparse_matrix_t, row::Clonglong,
                                               col::Clonglong,
                                               value::ComplexF64)::sparse_status_t
end

function mkl_sparse_s_update_values(A, nvalues, indx, indy, values)
    @ccall libmkl_rt.mkl_sparse_s_update_values(A::sparse_matrix_t, nvalues::BlasInt,
                                                indx::Ptr{BlasInt}, indy::Ptr{BlasInt},
                                                values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_update_values(A, nvalues, indx, indy, values)
    @ccall libmkl_rt.mkl_sparse_d_update_values(A::sparse_matrix_t, nvalues::BlasInt,
                                                indx::Ptr{BlasInt}, indy::Ptr{BlasInt},
                                                values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_update_values(A, nvalues, indx, indy, values)
    @ccall libmkl_rt.mkl_sparse_c_update_values(A::sparse_matrix_t, nvalues::BlasInt,
                                                indx::Ptr{BlasInt}, indy::Ptr{BlasInt},
                                                values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_update_values(A, nvalues, indx, indy, values)
    @ccall libmkl_rt.mkl_sparse_z_update_values(A::sparse_matrix_t, nvalues::BlasInt,
                                                indx::Ptr{BlasInt}, indy::Ptr{BlasInt},
                                                values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_update_values_64(A, nvalues, indx, indy, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_update_values_64(A::sparse_matrix_t, nvalues::Clonglong,
                                                   indx::Ptr{Clonglong},
                                                   indy::Ptr{Clonglong},
                                                   values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_update_values_64(A, nvalues, indx, indy, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_update_values_64(A::sparse_matrix_t, nvalues::Clonglong,
                                                   indx::Ptr{Clonglong},
                                                   indy::Ptr{Clonglong},
                                                   values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_update_values_64(A, nvalues, indx, indy, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_update_values_64(A::sparse_matrix_t, nvalues::Clonglong,
                                                   indx::Ptr{Clonglong},
                                                   indy::Ptr{Clonglong},
                                                   values::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_update_values_64(A, nvalues, indx, indy, values)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_update_values_64(A::sparse_matrix_t, nvalues::Clonglong,
                                                   indx::Ptr{Clonglong},
                                                   indy::Ptr{Clonglong},
                                                   values::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_set_verbose_mode(verbose)
    @ccall libmkl_rt.mkl_sparse_set_verbose_mode(verbose::verbose_mode_t)::sparse_status_t
end

function mkl_sparse_set_verbose_mode_64(verbose)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_verbose_mode_64(verbose::verbose_mode_t)::sparse_status_t
end

function mkl_sparse_set_mv_hint(A, operation, descr, expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_mv_hint(A::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            descr::matrix_descr,
                                            expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_mv_hint_64(A, operation, descr, expected_calls)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_mv_hint_64(A::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               descr::matrix_descr,
                                               expected_calls::Clonglong)::sparse_status_t
end

function mkl_sparse_set_dotmv_hint(A, operation, descr, expectedCalls)
    @ccall libmkl_rt.mkl_sparse_set_dotmv_hint(A::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               descr::matrix_descr,
                                               expectedCalls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_dotmv_hint_64(A, operation, descr, expectedCalls)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_dotmv_hint_64(A::sparse_matrix_t,
                                                  operation::sparse_operation_t,
                                                  descr::matrix_descr,
                                                  expectedCalls::Clonglong)::sparse_status_t
end

function mkl_sparse_set_mm_hint(A, operation, descr, layout, dense_matrix_size,
                                expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_mm_hint(A::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            descr::matrix_descr, layout::sparse_layout_t,
                                            dense_matrix_size::BlasInt,
                                            expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_mm_hint_64(A, operation, descr, layout, dense_matrix_size,
                                   expected_calls)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_mm_hint_64(A::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               descr::matrix_descr, layout::sparse_layout_t,
                                               dense_matrix_size::Clonglong,
                                               expected_calls::Clonglong)::sparse_status_t
end

function mkl_sparse_set_sv_hint(A, operation, descr, expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_sv_hint(A::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            descr::matrix_descr,
                                            expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_sv_hint_64(A, operation, descr, expected_calls)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_sv_hint_64(A::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               descr::matrix_descr,
                                               expected_calls::Clonglong)::sparse_status_t
end

function mkl_sparse_set_sm_hint(A, operation, descr, layout, dense_matrix_size,
                                expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_sm_hint(A::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            descr::matrix_descr, layout::sparse_layout_t,
                                            dense_matrix_size::BlasInt,
                                            expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_sm_hint_64(A, operation, descr, layout, dense_matrix_size,
                                   expected_calls)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_sm_hint_64(A::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               descr::matrix_descr, layout::sparse_layout_t,
                                               dense_matrix_size::Clonglong,
                                               expected_calls::Clonglong)::sparse_status_t
end

function mkl_sparse_set_symgs_hint(A, operation, descr, expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_symgs_hint(A::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               descr::matrix_descr,
                                               expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_symgs_hint_64(A, operation, descr, expected_calls)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_symgs_hint_64(A::sparse_matrix_t,
                                                  operation::sparse_operation_t,
                                                  descr::matrix_descr,
                                                  expected_calls::Clonglong)::sparse_status_t
end

function mkl_sparse_set_lu_smoother_hint(A, operation, descr, expectedCalls)
    @ccall libmkl_rt.mkl_sparse_set_lu_smoother_hint(A::sparse_matrix_t,
                                                     operation::sparse_operation_t,
                                                     descr::matrix_descr,
                                                     expectedCalls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_lu_smoother_hint_64(A, operation, descr, expectedCalls)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_lu_smoother_hint_64(A::sparse_matrix_t,
                                                        operation::sparse_operation_t,
                                                        descr::matrix_descr,
                                                        expectedCalls::Clonglong)::sparse_status_t
end

function mkl_sparse_set_sorv_hint(type, A, descr, expectedCalls)
    @ccall libmkl_rt.mkl_sparse_set_sorv_hint(type::sparse_sor_type_t, A::sparse_matrix_t,
                                              descr::matrix_descr,
                                              expectedCalls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_sorv_hint_64(type, A, descr, expectedCalls)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_sorv_hint_64(type::sparse_sor_type_t,
                                                 A::sparse_matrix_t, descr::matrix_descr,
                                                 expectedCalls::Clonglong)::sparse_status_t
end

function mkl_sparse_set_memory_hint(A, policy)
    @ccall libmkl_rt.mkl_sparse_set_memory_hint(A::sparse_matrix_t,
                                                policy::sparse_memory_usage_t)::sparse_status_t
end

function mkl_sparse_set_memory_hint_64(A, policy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_set_memory_hint_64(A::sparse_matrix_t,
                                                   policy::sparse_memory_usage_t)::sparse_status_t
end

function mkl_sparse_optimize(A)
    @ccall libmkl_rt.mkl_sparse_optimize(A::sparse_matrix_t)::sparse_status_t
end

function mkl_sparse_optimize_64(A)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_optimize_64(A::sparse_matrix_t)::sparse_status_t
end

function mkl_sparse_order(A)
    @ccall libmkl_rt.mkl_sparse_order(A::sparse_matrix_t)::sparse_status_t
end

function mkl_sparse_order_64(A)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_order_64(A::sparse_matrix_t)::sparse_status_t
end

function mkl_sparse_s_mv(operation, alpha, A, descr, x, beta, y)
    @ccall libmkl_rt.mkl_sparse_s_mv(operation::sparse_operation_t, alpha::Float32,
                                     A::sparse_matrix_t, descr::matrix_descr,
                                     x::Ptr{Float32}, beta::Float32,
                                     y::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_mv(operation, alpha, A, descr, x, beta, y)
    @ccall libmkl_rt.mkl_sparse_d_mv(operation::sparse_operation_t, alpha::Float64,
                                     A::sparse_matrix_t, descr::matrix_descr,
                                     x::Ptr{Float64}, beta::Float64,
                                     y::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_mv(operation, alpha, A, descr, x, beta, y)
    @ccall libmkl_rt.mkl_sparse_c_mv(operation::sparse_operation_t, alpha::ComplexF32,
                                     A::sparse_matrix_t, descr::matrix_descr,
                                     x::Ptr{ComplexF32}, beta::ComplexF32,
                                     y::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_mv(operation, alpha, A, descr, x, beta, y)
    @ccall libmkl_rt.mkl_sparse_z_mv(operation::sparse_operation_t, alpha::ComplexF64,
                                     A::sparse_matrix_t, descr::matrix_descr,
                                     x::Ptr{ComplexF64}, beta::ComplexF64,
                                     y::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_mv_64(operation, alpha, A, descr, x, beta, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_mv_64(operation::sparse_operation_t, alpha::Float32,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        x::Ptr{Float32}, beta::Float32,
                                        y::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_mv_64(operation, alpha, A, descr, x, beta, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_mv_64(operation::sparse_operation_t, alpha::Float64,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        x::Ptr{Float64}, beta::Float64,
                                        y::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_mv_64(operation, alpha, A, descr, x, beta, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_mv_64(operation::sparse_operation_t, alpha::ComplexF32,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        x::Ptr{ComplexF32}, beta::ComplexF32,
                                        y::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_mv_64(operation, alpha, A, descr, x, beta, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_mv_64(operation::sparse_operation_t, alpha::ComplexF64,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        x::Ptr{ComplexF64}, beta::ComplexF64,
                                        y::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_dotmv(transA, alpha, A, descr, x, beta, y, d)
    @ccall libmkl_rt.mkl_sparse_s_dotmv(transA::sparse_operation_t, alpha::Float32,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        x::Ptr{Float32}, beta::Float32, y::Ptr{Float32},
                                        d::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_dotmv(transA, alpha, A, descr, x, beta, y, d)
    @ccall libmkl_rt.mkl_sparse_d_dotmv(transA::sparse_operation_t, alpha::Float64,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        x::Ptr{Float64}, beta::Float64, y::Ptr{Float64},
                                        d::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_dotmv(transA, alpha, A, descr, x, beta, y, d)
    @ccall libmkl_rt.mkl_sparse_c_dotmv(transA::sparse_operation_t, alpha::ComplexF32,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        x::Ptr{ComplexF32}, beta::ComplexF32,
                                        y::Ptr{ComplexF32},
                                        d::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_dotmv(transA, alpha, A, descr, x, beta, y, d)
    @ccall libmkl_rt.mkl_sparse_z_dotmv(transA::sparse_operation_t, alpha::ComplexF64,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        x::Ptr{ComplexF64}, beta::ComplexF64,
                                        y::Ptr{ComplexF64},
                                        d::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_dotmv_64(transA, alpha, A, descr, x, beta, y, d)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_dotmv_64(transA::sparse_operation_t, alpha::Float32,
                                           A::sparse_matrix_t, descr::matrix_descr,
                                           x::Ptr{Float32}, beta::Float32, y::Ptr{Float32},
                                           d::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_dotmv_64(transA, alpha, A, descr, x, beta, y, d)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_dotmv_64(transA::sparse_operation_t, alpha::Float64,
                                           A::sparse_matrix_t, descr::matrix_descr,
                                           x::Ptr{Float64}, beta::Float64, y::Ptr{Float64},
                                           d::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_dotmv_64(transA, alpha, A, descr, x, beta, y, d)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_dotmv_64(transA::sparse_operation_t, alpha::ComplexF32,
                                           A::sparse_matrix_t, descr::matrix_descr,
                                           x::Ptr{ComplexF32}, beta::ComplexF32,
                                           y::Ptr{ComplexF32},
                                           d::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_dotmv_64(transA, alpha, A, descr, x, beta, y, d)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_dotmv_64(transA::sparse_operation_t, alpha::ComplexF64,
                                           A::sparse_matrix_t, descr::matrix_descr,
                                           x::Ptr{ComplexF64}, beta::ComplexF64,
                                           y::Ptr{ComplexF64},
                                           d::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_trsv(operation, alpha, A, descr, x, y)
    @ccall libmkl_rt.mkl_sparse_s_trsv(operation::sparse_operation_t, alpha::Float32,
                                       A::sparse_matrix_t, descr::matrix_descr,
                                       x::Ptr{Float32}, y::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_trsv(operation, alpha, A, descr, x, y)
    @ccall libmkl_rt.mkl_sparse_d_trsv(operation::sparse_operation_t, alpha::Float64,
                                       A::sparse_matrix_t, descr::matrix_descr,
                                       x::Ptr{Float64}, y::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_trsv(operation, alpha, A, descr, x, y)
    @ccall libmkl_rt.mkl_sparse_c_trsv(operation::sparse_operation_t, alpha::ComplexF32,
                                       A::sparse_matrix_t, descr::matrix_descr,
                                       x::Ptr{ComplexF32},
                                       y::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_trsv(operation, alpha, A, descr, x, y)
    @ccall libmkl_rt.mkl_sparse_z_trsv(operation::sparse_operation_t, alpha::ComplexF64,
                                       A::sparse_matrix_t, descr::matrix_descr,
                                       x::Ptr{ComplexF64},
                                       y::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_trsv_64(operation, alpha, A, descr, x, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_trsv_64(operation::sparse_operation_t, alpha::Float32,
                                          A::sparse_matrix_t, descr::matrix_descr,
                                          x::Ptr{Float32}, y::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_trsv_64(operation, alpha, A, descr, x, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_trsv_64(operation::sparse_operation_t, alpha::Float64,
                                          A::sparse_matrix_t, descr::matrix_descr,
                                          x::Ptr{Float64}, y::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_trsv_64(operation, alpha, A, descr, x, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_trsv_64(operation::sparse_operation_t,
                                          alpha::ComplexF32, A::sparse_matrix_t,
                                          descr::matrix_descr, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_trsv_64(operation, alpha, A, descr, x, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_trsv_64(operation::sparse_operation_t,
                                          alpha::ComplexF64, A::sparse_matrix_t,
                                          descr::matrix_descr, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_symgs(op, A, descr, alpha, b, x)
    @ccall libmkl_rt.mkl_sparse_s_symgs(op::sparse_operation_t, A::sparse_matrix_t,
                                        descr::matrix_descr, alpha::Float32, b::Ptr{Float32},
                                        x::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_symgs(op, A, descr, alpha, b, x)
    @ccall libmkl_rt.mkl_sparse_d_symgs(op::sparse_operation_t, A::sparse_matrix_t,
                                        descr::matrix_descr, alpha::Float64,
                                        b::Ptr{Float64}, x::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_symgs(op, A, descr, alpha, b, x)
    @ccall libmkl_rt.mkl_sparse_c_symgs(op::sparse_operation_t, A::sparse_matrix_t,
                                        descr::matrix_descr, alpha::ComplexF32,
                                        b::Ptr{ComplexF32},
                                        x::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_symgs(op, A, descr, alpha, b, x)
    @ccall libmkl_rt.mkl_sparse_z_symgs(op::sparse_operation_t, A::sparse_matrix_t,
                                        descr::matrix_descr, alpha::ComplexF64,
                                        b::Ptr{ComplexF64},
                                        x::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_symgs_64(op, A, descr, alpha, b, x)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_symgs_64(op::sparse_operation_t, A::sparse_matrix_t,
                                           descr::matrix_descr, alpha::Float32,
                                           b::Ptr{Float32}, x::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_symgs_64(op, A, descr, alpha, b, x)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_symgs_64(op::sparse_operation_t, A::sparse_matrix_t,
                                           descr::matrix_descr, alpha::Float64,
                                           b::Ptr{Float64},
                                           x::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_symgs_64(op, A, descr, alpha, b, x)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_symgs_64(op::sparse_operation_t, A::sparse_matrix_t,
                                           descr::matrix_descr, alpha::ComplexF32,
                                           b::Ptr{ComplexF32},
                                           x::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_symgs_64(op, A, descr, alpha, b, x)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_symgs_64(op::sparse_operation_t, A::sparse_matrix_t,
                                           descr::matrix_descr, alpha::ComplexF64,
                                           b::Ptr{ComplexF64},
                                           x::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_symgs_mv(op, A, descr, alpha, b, x, y)
    @ccall libmkl_rt.mkl_sparse_s_symgs_mv(op::sparse_operation_t, A::sparse_matrix_t,
                                           descr::matrix_descr, alpha::Float32,
                                           b::Ptr{Float32}, x::Ptr{Float32},
                                           y::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_symgs_mv(op, A, descr, alpha, b, x, y)
    @ccall libmkl_rt.mkl_sparse_d_symgs_mv(op::sparse_operation_t, A::sparse_matrix_t,
                                           descr::matrix_descr, alpha::Float64,
                                           b::Ptr{Float64}, x::Ptr{Float64},
                                           y::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_symgs_mv(op, A, descr, alpha, b, x, y)
    @ccall libmkl_rt.mkl_sparse_c_symgs_mv(op::sparse_operation_t, A::sparse_matrix_t,
                                           descr::matrix_descr, alpha::ComplexF32,
                                           b::Ptr{ComplexF32}, x::Ptr{ComplexF32},
                                           y::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_symgs_mv(op, A, descr, alpha, b, x, y)
    @ccall libmkl_rt.mkl_sparse_z_symgs_mv(op::sparse_operation_t, A::sparse_matrix_t,
                                           descr::matrix_descr, alpha::ComplexF64,
                                           b::Ptr{ComplexF64}, x::Ptr{ComplexF64},
                                           y::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_symgs_mv_64(op, A, descr, alpha, b, x, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_symgs_mv_64(op::sparse_operation_t, A::sparse_matrix_t,
                                              descr::matrix_descr, alpha::Float32,
                                              b::Ptr{Float32}, x::Ptr{Float32},
                                              y::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_symgs_mv_64(op, A, descr, alpha, b, x, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_symgs_mv_64(op::sparse_operation_t, A::sparse_matrix_t,
                                              descr::matrix_descr, alpha::Float64,
                                              b::Ptr{Float64}, x::Ptr{Float64},
                                              y::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_symgs_mv_64(op, A, descr, alpha, b, x, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_symgs_mv_64(op::sparse_operation_t, A::sparse_matrix_t,
                                              descr::matrix_descr, alpha::ComplexF32,
                                              b::Ptr{ComplexF32}, x::Ptr{ComplexF32},
                                              y::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_symgs_mv_64(op, A, descr, alpha, b, x, y)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_symgs_mv_64(op::sparse_operation_t, A::sparse_matrix_t,
                                              descr::matrix_descr, alpha::ComplexF64,
                                              b::Ptr{ComplexF64}, x::Ptr{ComplexF64},
                                              y::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_lu_smoother(op, A, descr, diag, approx_diag_inverse, x, rhs)
    @ccall libmkl_rt.mkl_sparse_s_lu_smoother(op::sparse_operation_t, A::sparse_matrix_t,
                                              descr::matrix_descr, diag::Ptr{Float32},
                                              approx_diag_inverse::Ptr{Float32},
                                              x::Ptr{Float32},
                                              rhs::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_lu_smoother(op, A, descr, diag, approx_diag_inverse, x, rhs)
    @ccall libmkl_rt.mkl_sparse_d_lu_smoother(op::sparse_operation_t, A::sparse_matrix_t,
                                              descr::matrix_descr, diag::Ptr{Float64},
                                              approx_diag_inverse::Ptr{Float64},
                                              x::Ptr{Float64},
                                              rhs::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_lu_smoother(op, A, descr, diag, approx_diag_inverse, x, rhs)
    @ccall libmkl_rt.mkl_sparse_c_lu_smoother(op::sparse_operation_t, A::sparse_matrix_t,
                                              descr::matrix_descr, diag::Ptr{ComplexF32},
                                              approx_diag_inverse::Ptr{ComplexF32},
                                              x::Ptr{ComplexF32},
                                              rhs::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_lu_smoother(op, A, descr, diag, approx_diag_inverse, x, rhs)
    @ccall libmkl_rt.mkl_sparse_z_lu_smoother(op::sparse_operation_t, A::sparse_matrix_t,
                                              descr::matrix_descr, diag::Ptr{ComplexF64},
                                              approx_diag_inverse::Ptr{ComplexF64},
                                              x::Ptr{ComplexF64},
                                              rhs::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_lu_smoother_64(op, A, descr, diag, approx_diag_inverse, x, rhs)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_lu_smoother_64(op::sparse_operation_t, A::sparse_matrix_t,
                                                 descr::matrix_descr, diag::Ptr{Float32},
                                                 approx_diag_inverse::Ptr{Float32},
                                                 x::Ptr{Float32},
                                                 rhs::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_lu_smoother_64(op, A, descr, diag, approx_diag_inverse, x, rhs)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_lu_smoother_64(op::sparse_operation_t, A::sparse_matrix_t,
                                                 descr::matrix_descr, diag::Ptr{Float64},
                                                 approx_diag_inverse::Ptr{Float64},
                                                 x::Ptr{Float64},
                                                 rhs::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_c_lu_smoother_64(op, A, descr, diag, approx_diag_inverse, x, rhs)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_lu_smoother_64(op::sparse_operation_t, A::sparse_matrix_t,
                                                 descr::matrix_descr,
                                                 diag::Ptr{ComplexF32},
                                                 approx_diag_inverse::Ptr{ComplexF32},
                                                 x::Ptr{ComplexF32},
                                                 rhs::Ptr{ComplexF32})::sparse_status_t
end

function mkl_sparse_z_lu_smoother_64(op, A, descr, diag, approx_diag_inverse, x, rhs)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_lu_smoother_64(op::sparse_operation_t, A::sparse_matrix_t,
                                                 descr::matrix_descr,
                                                 diag::Ptr{ComplexF64},
                                                 approx_diag_inverse::Ptr{ComplexF64},
                                                 x::Ptr{ComplexF64},
                                                 rhs::Ptr{ComplexF64})::sparse_status_t
end

function mkl_sparse_s_mm(operation, alpha, A, descr, layout, x, columns, ldx, beta, y, ldy)
    @ccall libmkl_rt.mkl_sparse_s_mm(operation::sparse_operation_t, alpha::Float32,
                                     A::sparse_matrix_t, descr::matrix_descr,
                                     layout::sparse_layout_t, x::Ptr{Float32}, columns::BlasInt,
                                     ldx::BlasInt, beta::Float32, y::Ptr{Float32},
                                     ldy::BlasInt)::sparse_status_t
end

function mkl_sparse_d_mm(operation, alpha, A, descr, layout, x, columns, ldx, beta, y, ldy)
    @ccall libmkl_rt.mkl_sparse_d_mm(operation::sparse_operation_t, alpha::Float64,
                                     A::sparse_matrix_t, descr::matrix_descr,
                                     layout::sparse_layout_t, x::Ptr{Float64},
                                     columns::BlasInt, ldx::BlasInt, beta::Float64,
                                     y::Ptr{Float64}, ldy::BlasInt)::sparse_status_t
end

function mkl_sparse_c_mm(operation, alpha, A, descr, layout, x, columns, ldx, beta, y, ldy)
    @ccall libmkl_rt.mkl_sparse_c_mm(operation::sparse_operation_t, alpha::ComplexF32,
                                     A::sparse_matrix_t, descr::matrix_descr,
                                     layout::sparse_layout_t, x::Ptr{ComplexF32},
                                     columns::BlasInt, ldx::BlasInt, beta::ComplexF32,
                                     y::Ptr{ComplexF32}, ldy::BlasInt)::sparse_status_t
end

function mkl_sparse_z_mm(operation, alpha, A, descr, layout, x, columns, ldx, beta, y, ldy)
    @ccall libmkl_rt.mkl_sparse_z_mm(operation::sparse_operation_t, alpha::ComplexF64,
                                     A::sparse_matrix_t, descr::matrix_descr,
                                     layout::sparse_layout_t, x::Ptr{ComplexF64},
                                     columns::BlasInt, ldx::BlasInt, beta::ComplexF64,
                                     y::Ptr{ComplexF64}, ldy::BlasInt)::sparse_status_t
end

function mkl_sparse_s_mm_64(operation, alpha, A, descr, layout, x, columns, ldx, beta, y,
                            ldy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_mm_64(operation::sparse_operation_t, alpha::Float32,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        layout::sparse_layout_t, x::Ptr{Float32},
                                        columns::Clonglong, ldx::Clonglong, beta::Float32,
                                        y::Ptr{Float32}, ldy::Clonglong)::sparse_status_t
end

function mkl_sparse_d_mm_64(operation, alpha, A, descr, layout, x, columns, ldx, beta, y,
                            ldy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_mm_64(operation::sparse_operation_t, alpha::Float64,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        layout::sparse_layout_t, x::Ptr{Float64},
                                        columns::Clonglong, ldx::Clonglong, beta::Float64,
                                        y::Ptr{Float64}, ldy::Clonglong)::sparse_status_t
end

function mkl_sparse_c_mm_64(operation, alpha, A, descr, layout, x, columns, ldx, beta, y,
                            ldy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_mm_64(operation::sparse_operation_t, alpha::ComplexF32,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        layout::sparse_layout_t, x::Ptr{ComplexF32},
                                        columns::Clonglong, ldx::Clonglong,
                                        beta::ComplexF32, y::Ptr{ComplexF32},
                                        ldy::Clonglong)::sparse_status_t
end

function mkl_sparse_z_mm_64(operation, alpha, A, descr, layout, x, columns, ldx, beta, y,
                            ldy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_mm_64(operation::sparse_operation_t, alpha::ComplexF64,
                                        A::sparse_matrix_t, descr::matrix_descr,
                                        layout::sparse_layout_t, x::Ptr{ComplexF64},
                                        columns::Clonglong, ldx::Clonglong,
                                        beta::ComplexF64, y::Ptr{ComplexF64},
                                        ldy::Clonglong)::sparse_status_t
end

function mkl_sparse_s_trsm(operation, alpha, A, descr, layout, x, columns, ldx, y, ldy)
    @ccall libmkl_rt.mkl_sparse_s_trsm(operation::sparse_operation_t, alpha::Float32,
                                       A::sparse_matrix_t, descr::matrix_descr,
                                       layout::sparse_layout_t, x::Ptr{Float32},
                                       columns::BlasInt, ldx::BlasInt, y::Ptr{Float32},
                                       ldy::BlasInt)::sparse_status_t
end

function mkl_sparse_d_trsm(operation, alpha, A, descr, layout, x, columns, ldx, y, ldy)
    @ccall libmkl_rt.mkl_sparse_d_trsm(operation::sparse_operation_t, alpha::Float64,
                                       A::sparse_matrix_t, descr::matrix_descr,
                                       layout::sparse_layout_t, x::Ptr{Float64},
                                       columns::BlasInt, ldx::BlasInt, y::Ptr{Float64},
                                       ldy::BlasInt)::sparse_status_t
end

function mkl_sparse_c_trsm(operation, alpha, A, descr, layout, x, columns, ldx, y, ldy)
    @ccall libmkl_rt.mkl_sparse_c_trsm(operation::sparse_operation_t, alpha::ComplexF32,
                                       A::sparse_matrix_t, descr::matrix_descr,
                                       layout::sparse_layout_t, x::Ptr{ComplexF32},
                                       columns::BlasInt, ldx::BlasInt, y::Ptr{ComplexF32},
                                       ldy::BlasInt)::sparse_status_t
end

function mkl_sparse_z_trsm(operation, alpha, A, descr, layout, x, columns, ldx, y, ldy)
    @ccall libmkl_rt.mkl_sparse_z_trsm(operation::sparse_operation_t, alpha::ComplexF64,
                                       A::sparse_matrix_t, descr::matrix_descr,
                                       layout::sparse_layout_t, x::Ptr{ComplexF64},
                                       columns::BlasInt, ldx::BlasInt, y::Ptr{ComplexF64},
                                       ldy::BlasInt)::sparse_status_t
end

function mkl_sparse_s_trsm_64(operation, alpha, A, descr, layout, x, columns, ldx, y, ldy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_trsm_64(operation::sparse_operation_t, alpha::Float32,
                                          A::sparse_matrix_t, descr::matrix_descr,
                                          layout::sparse_layout_t, x::Ptr{Float32},
                                          columns::Clonglong, ldx::Clonglong,
                                          y::Ptr{Float32}, ldy::Clonglong)::sparse_status_t
end

function mkl_sparse_d_trsm_64(operation, alpha, A, descr, layout, x, columns, ldx, y, ldy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_trsm_64(operation::sparse_operation_t, alpha::Float64,
                                          A::sparse_matrix_t, descr::matrix_descr,
                                          layout::sparse_layout_t, x::Ptr{Float64},
                                          columns::Clonglong, ldx::Clonglong,
                                          y::Ptr{Float64}, ldy::Clonglong)::sparse_status_t
end

function mkl_sparse_c_trsm_64(operation, alpha, A, descr, layout, x, columns, ldx, y, ldy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_trsm_64(operation::sparse_operation_t,
                                          alpha::ComplexF32, A::sparse_matrix_t,
                                          descr::matrix_descr, layout::sparse_layout_t,
                                          x::Ptr{ComplexF32}, columns::Clonglong,
                                          ldx::Clonglong, y::Ptr{ComplexF32},
                                          ldy::Clonglong)::sparse_status_t
end

function mkl_sparse_z_trsm_64(operation, alpha, A, descr, layout, x, columns, ldx, y, ldy)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_trsm_64(operation::sparse_operation_t,
                                          alpha::ComplexF64, A::sparse_matrix_t,
                                          descr::matrix_descr, layout::sparse_layout_t,
                                          x::Ptr{ComplexF64}, columns::Clonglong,
                                          ldx::Clonglong, y::Ptr{ComplexF64},
                                          ldy::Clonglong)::sparse_status_t
end

function mkl_sparse_s_add(operation, A, alpha, B, C)
    @ccall libmkl_rt.mkl_sparse_s_add(operation::sparse_operation_t, A::sparse_matrix_t,
                                      alpha::Float32, B::sparse_matrix_t,
                                      C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_d_add(operation, A, alpha, B, C)
    @ccall libmkl_rt.mkl_sparse_d_add(operation::sparse_operation_t, A::sparse_matrix_t,
                                      alpha::Float64, B::sparse_matrix_t,
                                      C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_c_add(operation, A, alpha, B, C)
    @ccall libmkl_rt.mkl_sparse_c_add(operation::sparse_operation_t, A::sparse_matrix_t,
                                      alpha::ComplexF32, B::sparse_matrix_t,
                                      C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_z_add(operation, A, alpha, B, C)
    @ccall libmkl_rt.mkl_sparse_z_add(operation::sparse_operation_t, A::sparse_matrix_t,
                                      alpha::ComplexF64, B::sparse_matrix_t,
                                      C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_s_add_64(operation, A, alpha, B, C)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_add_64(operation::sparse_operation_t, A::sparse_matrix_t,
                                         alpha::Float32, B::sparse_matrix_t,
                                         C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_d_add_64(operation, A, alpha, B, C)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_add_64(operation::sparse_operation_t, A::sparse_matrix_t,
                                         alpha::Float64, B::sparse_matrix_t,
                                         C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_c_add_64(operation, A, alpha, B, C)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_add_64(operation::sparse_operation_t, A::sparse_matrix_t,
                                         alpha::ComplexF32, B::sparse_matrix_t,
                                         C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_z_add_64(operation, A, alpha, B, C)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_add_64(operation::sparse_operation_t, A::sparse_matrix_t,
                                         alpha::ComplexF64, B::sparse_matrix_t,
                                         C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_spmm(operation, A, B, C)
    @ccall libmkl_rt.mkl_sparse_spmm(operation::sparse_operation_t, A::sparse_matrix_t,
                                     B::sparse_matrix_t,
                                     C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_spmm_64(operation, A, B, C)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_spmm_64(operation::sparse_operation_t, A::sparse_matrix_t,
                                        B::sparse_matrix_t,
                                        C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_sp2m(transA, descrA, A, transB, descrB, B, request, C)
    @ccall libmkl_rt.mkl_sparse_sp2m(transA::sparse_operation_t, descrA::matrix_descr,
                                     A::sparse_matrix_t, transB::sparse_operation_t,
                                     descrB::matrix_descr, B::sparse_matrix_t,
                                     request::sparse_request_t,
                                     C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_sp2m_64(transA, descrA, A, transB, descrB, B, request, C)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_sp2m_64(transA::sparse_operation_t, descrA::matrix_descr,
                                        A::sparse_matrix_t, transB::sparse_operation_t,
                                        descrB::matrix_descr, B::sparse_matrix_t,
                                        request::sparse_request_t,
                                        C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_syrk(operation, A, C)
    @ccall libmkl_rt.mkl_sparse_syrk(operation::sparse_operation_t, A::sparse_matrix_t,
                                     C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_syrk_64(operation, A, C)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_syrk_64(operation::sparse_operation_t, A::sparse_matrix_t,
                                        C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_sypr(transA, A, B, descrB, C, request)
    @ccall libmkl_rt.mkl_sparse_sypr(transA::sparse_operation_t, A::sparse_matrix_t,
                                     B::sparse_matrix_t, descrB::matrix_descr,
                                     C::Ptr{sparse_matrix_t},
                                     request::sparse_request_t)::sparse_status_t
end

function mkl_sparse_sypr_64(transA, A, B, descrB, C, request)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_sypr_64(transA::sparse_operation_t, A::sparse_matrix_t,
                                        B::sparse_matrix_t, descrB::matrix_descr,
                                        C::Ptr{sparse_matrix_t},
                                        request::sparse_request_t)::sparse_status_t
end

function mkl_sparse_s_syprd(op, A, B, layoutB, ldb, alpha, beta, C, layoutC, ldc)
    @ccall libmkl_rt.mkl_sparse_s_syprd(op::sparse_operation_t, A::sparse_matrix_t,
                                        B::Ptr{Float32}, layoutB::sparse_layout_t, ldb::BlasInt,
                                        alpha::Float32, beta::Float32, C::Ptr{Float32},
                                        layoutC::sparse_layout_t,
                                        ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_d_syprd(op, A, B, layoutB, ldb, alpha, beta, C, layoutC, ldc)
    @ccall libmkl_rt.mkl_sparse_d_syprd(op::sparse_operation_t, A::sparse_matrix_t,
                                        B::Ptr{Float64}, layoutB::sparse_layout_t,
                                        ldb::BlasInt, alpha::Float64, beta::Float64,
                                        C::Ptr{Float64}, layoutC::sparse_layout_t,
                                        ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_c_syprd(op, A, B, layoutB, ldb, alpha, beta, C, layoutC, ldc)
    @ccall libmkl_rt.mkl_sparse_c_syprd(op::sparse_operation_t, A::sparse_matrix_t,
                                        B::Ptr{ComplexF32}, layoutB::sparse_layout_t,
                                        ldb::BlasInt, alpha::ComplexF32, beta::ComplexF32,
                                        C::Ptr{ComplexF32}, layoutC::sparse_layout_t,
                                        ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_z_syprd(op, A, B, layoutB, ldb, alpha, beta, C, layoutC, ldc)
    @ccall libmkl_rt.mkl_sparse_z_syprd(op::sparse_operation_t, A::sparse_matrix_t,
                                        B::Ptr{ComplexF64}, layoutB::sparse_layout_t,
                                        ldb::BlasInt, alpha::ComplexF64,
                                        beta::ComplexF64, C::Ptr{ComplexF64},
                                        layoutC::sparse_layout_t,
                                        ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_s_syprd_64(op, A, B, layoutB, ldb, alpha, beta, C, layoutC, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_syprd_64(op::sparse_operation_t, A::sparse_matrix_t,
                                           B::Ptr{Float32}, layoutB::sparse_layout_t,
                                           ldb::Clonglong, alpha::Float32, beta::Float32,
                                           C::Ptr{Float32}, layoutC::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_d_syprd_64(op, A, B, layoutB, ldb, alpha, beta, C, layoutC, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_syprd_64(op::sparse_operation_t, A::sparse_matrix_t,
                                           B::Ptr{Float64}, layoutB::sparse_layout_t,
                                           ldb::Clonglong, alpha::Float64, beta::Float64,
                                           C::Ptr{Float64}, layoutC::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_c_syprd_64(op, A, B, layoutB, ldb, alpha, beta, C, layoutC, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_syprd_64(op::sparse_operation_t, A::sparse_matrix_t,
                                           B::Ptr{ComplexF32}, layoutB::sparse_layout_t,
                                           ldb::Clonglong, alpha::ComplexF32,
                                           beta::ComplexF32, C::Ptr{ComplexF32},
                                           layoutC::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_z_syprd_64(op, A, B, layoutB, ldb, alpha, beta, C, layoutC, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_syprd_64(op::sparse_operation_t, A::sparse_matrix_t,
                                           B::Ptr{ComplexF64}, layoutB::sparse_layout_t,
                                           ldb::Clonglong, alpha::ComplexF64,
                                           beta::ComplexF64, C::Ptr{ComplexF64},
                                           layoutC::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_s_spmmd(operation, A, B, layout, C, ldc)
    @ccall libmkl_rt.mkl_sparse_s_spmmd(operation::sparse_operation_t, A::sparse_matrix_t,
                                        B::sparse_matrix_t, layout::sparse_layout_t,
                                        C::Ptr{Float32}, ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_d_spmmd(operation, A, B, layout, C, ldc)
    @ccall libmkl_rt.mkl_sparse_d_spmmd(operation::sparse_operation_t, A::sparse_matrix_t,
                                        B::sparse_matrix_t, layout::sparse_layout_t,
                                        C::Ptr{Float64}, ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_c_spmmd(operation, A, B, layout, C, ldc)
    @ccall libmkl_rt.mkl_sparse_c_spmmd(operation::sparse_operation_t, A::sparse_matrix_t,
                                        B::sparse_matrix_t, layout::sparse_layout_t,
                                        C::Ptr{ComplexF32}, ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_z_spmmd(operation, A, B, layout, C, ldc)
    @ccall libmkl_rt.mkl_sparse_z_spmmd(operation::sparse_operation_t, A::sparse_matrix_t,
                                        B::sparse_matrix_t, layout::sparse_layout_t,
                                        C::Ptr{ComplexF64}, ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_s_spmmd_64(operation, A, B, layout, C, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_spmmd_64(operation::sparse_operation_t,
                                           A::sparse_matrix_t, B::sparse_matrix_t,
                                           layout::sparse_layout_t, C::Ptr{Float32},
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_d_spmmd_64(operation, A, B, layout, C, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_spmmd_64(operation::sparse_operation_t,
                                           A::sparse_matrix_t, B::sparse_matrix_t,
                                           layout::sparse_layout_t, C::Ptr{Float64},
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_c_spmmd_64(operation, A, B, layout, C, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_spmmd_64(operation::sparse_operation_t,
                                           A::sparse_matrix_t, B::sparse_matrix_t,
                                           layout::sparse_layout_t, C::Ptr{ComplexF32},
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_z_spmmd_64(operation, A, B, layout, C, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_spmmd_64(operation::sparse_operation_t,
                                           A::sparse_matrix_t, B::sparse_matrix_t,
                                           layout::sparse_layout_t, C::Ptr{ComplexF64},
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_s_sp2md(transA, descrA, A, transB, descrB, B, alpha, beta, C, layout,
                            ldc)
    @ccall libmkl_rt.mkl_sparse_s_sp2md(transA::sparse_operation_t, descrA::matrix_descr,
                                        A::sparse_matrix_t, transB::sparse_operation_t,
                                        descrB::matrix_descr, B::sparse_matrix_t,
                                        alpha::Float32, beta::Float32, C::Ptr{Float32},
                                        layout::sparse_layout_t, ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_d_sp2md(transA, descrA, A, transB, descrB, B, alpha, beta, C, layout,
                            ldc)
    @ccall libmkl_rt.mkl_sparse_d_sp2md(transA::sparse_operation_t, descrA::matrix_descr,
                                        A::sparse_matrix_t, transB::sparse_operation_t,
                                        descrB::matrix_descr, B::sparse_matrix_t,
                                        alpha::Float64, beta::Float64, C::Ptr{Float64},
                                        layout::sparse_layout_t, ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_c_sp2md(transA, descrA, A, transB, descrB, B, alpha, beta, C, layout,
                            ldc)
    @ccall libmkl_rt.mkl_sparse_c_sp2md(transA::sparse_operation_t, descrA::matrix_descr,
                                        A::sparse_matrix_t, transB::sparse_operation_t,
                                        descrB::matrix_descr, B::sparse_matrix_t,
                                        alpha::ComplexF32, beta::ComplexF32,
                                        C::Ptr{ComplexF32}, layout::sparse_layout_t,
                                        ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_z_sp2md(transA, descrA, A, transB, descrB, B, alpha, beta, C, layout,
                            ldc)
    @ccall libmkl_rt.mkl_sparse_z_sp2md(transA::sparse_operation_t, descrA::matrix_descr,
                                        A::sparse_matrix_t, transB::sparse_operation_t,
                                        descrB::matrix_descr, B::sparse_matrix_t,
                                        alpha::ComplexF64, beta::ComplexF64,
                                        C::Ptr{ComplexF64}, layout::sparse_layout_t,
                                        ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_s_sp2md_64(transA, descrA, A, transB, descrB, B, alpha, beta, C, layout,
                               ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_sp2md_64(transA::sparse_operation_t, descrA::matrix_descr,
                                           A::sparse_matrix_t, transB::sparse_operation_t,
                                           descrB::matrix_descr, B::sparse_matrix_t,
                                           alpha::Float32, beta::Float32, C::Ptr{Float32},
                                           layout::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_d_sp2md_64(transA, descrA, A, transB, descrB, B, alpha, beta, C, layout,
                               ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_sp2md_64(transA::sparse_operation_t, descrA::matrix_descr,
                                           A::sparse_matrix_t, transB::sparse_operation_t,
                                           descrB::matrix_descr, B::sparse_matrix_t,
                                           alpha::Float64, beta::Float64, C::Ptr{Float64},
                                           layout::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_c_sp2md_64(transA, descrA, A, transB, descrB, B, alpha, beta, C, layout,
                               ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_sp2md_64(transA::sparse_operation_t, descrA::matrix_descr,
                                           A::sparse_matrix_t, transB::sparse_operation_t,
                                           descrB::matrix_descr, B::sparse_matrix_t,
                                           alpha::ComplexF32, beta::ComplexF32,
                                           C::Ptr{ComplexF32}, layout::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_z_sp2md_64(transA, descrA, A, transB, descrB, B, alpha, beta, C, layout,
                               ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_sp2md_64(transA::sparse_operation_t, descrA::matrix_descr,
                                           A::sparse_matrix_t, transB::sparse_operation_t,
                                           descrB::matrix_descr, B::sparse_matrix_t,
                                           alpha::ComplexF64, beta::ComplexF64,
                                           C::Ptr{ComplexF64}, layout::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_s_syrkd(operation, A, alpha, beta, C, layout, ldc)
    @ccall libmkl_rt.mkl_sparse_s_syrkd(operation::sparse_operation_t, A::sparse_matrix_t,
                                        alpha::Float32, beta::Float32, C::Ptr{Float32},
                                        layout::sparse_layout_t, ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_d_syrkd(operation, A, alpha, beta, C, layout, ldc)
    @ccall libmkl_rt.mkl_sparse_d_syrkd(operation::sparse_operation_t, A::sparse_matrix_t,
                                        alpha::Float64, beta::Float64, C::Ptr{Float64},
                                        layout::sparse_layout_t, ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_c_syrkd(operation, A, alpha, beta, C, layout, ldc)
    @ccall libmkl_rt.mkl_sparse_c_syrkd(operation::sparse_operation_t, A::sparse_matrix_t,
                                        alpha::ComplexF32, beta::ComplexF32,
                                        C::Ptr{ComplexF32}, layout::sparse_layout_t,
                                        ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_z_syrkd(operation, A, alpha, beta, C, layout, ldc)
    @ccall libmkl_rt.mkl_sparse_z_syrkd(operation::sparse_operation_t, A::sparse_matrix_t,
                                        alpha::ComplexF64, beta::ComplexF64,
                                        C::Ptr{ComplexF64}, layout::sparse_layout_t,
                                        ldc::BlasInt)::sparse_status_t
end

function mkl_sparse_s_syrkd_64(operation, A, alpha, beta, C, layout, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_syrkd_64(operation::sparse_operation_t,
                                           A::sparse_matrix_t, alpha::Float32, beta::Float32,
                                           C::Ptr{Float32}, layout::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_d_syrkd_64(operation, A, alpha, beta, C, layout, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_syrkd_64(operation::sparse_operation_t,
                                           A::sparse_matrix_t, alpha::Float64,
                                           beta::Float64, C::Ptr{Float64},
                                           layout::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_c_syrkd_64(operation, A, alpha, beta, C, layout, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_c_syrkd_64(operation::sparse_operation_t,
                                           A::sparse_matrix_t, alpha::ComplexF32,
                                           beta::ComplexF32, C::Ptr{ComplexF32},
                                           layout::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_z_syrkd_64(operation, A, alpha, beta, C, layout, ldc)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_z_syrkd_64(operation::sparse_operation_t,
                                           A::sparse_matrix_t, alpha::ComplexF64,
                                           beta::ComplexF64, C::Ptr{ComplexF64},
                                           layout::sparse_layout_t,
                                           ldc::Clonglong)::sparse_status_t
end

function mkl_sparse_s_sorv(type, descrA, A, omega, alpha, x, b)
    @ccall libmkl_rt.mkl_sparse_s_sorv(type::sparse_sor_type_t, descrA::matrix_descr,
                                       A::sparse_matrix_t, omega::Float32, alpha::Float32,
                                       x::Ptr{Float32}, b::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_sorv(type, descrA, A, omega, alpha, x, b)
    @ccall libmkl_rt.mkl_sparse_d_sorv(type::sparse_sor_type_t, descrA::matrix_descr,
                                       A::sparse_matrix_t, omega::Float64, alpha::Float64,
                                       x::Ptr{Float64}, b::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_s_sorv_64(type, descrA, A, omega, alpha, x, b)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_s_sorv_64(type::sparse_sor_type_t, descrA::matrix_descr,
                                          A::sparse_matrix_t, omega::Float32, alpha::Float32,
                                          x::Ptr{Float32}, b::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_sorv_64(type, descrA, A, omega, alpha, x, b)
    Sys.isapple() && error("This routine is not supported on Mac.")
    @ccall libmkl_rt.mkl_sparse_d_sorv_64(type::sparse_sor_type_t, descrA::matrix_descr,
                                          A::sparse_matrix_t, omega::Float64,
                                          alpha::Float64, x::Ptr{Float64},
                                          b::Ptr{Float64})::sparse_status_t
end

@enum sparse_qr_hint_t::UInt32 begin
    SPARSE_QR_WITH_PIVOTS = 0
end

function mkl_sparse_set_qr_hint(A, hint)
    @ccall libmkl_rt.mkl_sparse_set_qr_hint(A::sparse_matrix_t,
                                            hint::sparse_qr_hint_t)::sparse_status_t
end

function mkl_sparse_d_qr(operation, A, descr, layout, columns, x, ldx, b, ldb)
    @ccall libmkl_rt.mkl_sparse_d_qr(operation::sparse_operation_t, A::sparse_matrix_t,
                                     descr::matrix_descr, layout::sparse_layout_t,
                                     columns::BlasInt, x::Ptr{Float64}, ldx::BlasInt,
                                     b::Ptr{Float64}, ldb::BlasInt)::sparse_status_t
end

function mkl_sparse_s_qr(operation, A, descr, layout, columns, x, ldx, b, ldb)
    @ccall libmkl_rt.mkl_sparse_s_qr(operation::sparse_operation_t, A::sparse_matrix_t,
                                     descr::matrix_descr, layout::sparse_layout_t,
                                     columns::BlasInt, x::Ptr{Float32}, ldx::BlasInt,
                                     b::Ptr{Float32}, ldb::BlasInt)::sparse_status_t
end

function mkl_sparse_qr_reorder(A, descr)
    @ccall libmkl_rt.mkl_sparse_qr_reorder(A::sparse_matrix_t,
                                           descr::matrix_descr)::sparse_status_t
end

function mkl_sparse_d_qr_factorize(A, alt_values)
    @ccall libmkl_rt.mkl_sparse_d_qr_factorize(A::sparse_matrix_t,
                                               alt_values::Ptr{Float64})::sparse_status_t
end

function mkl_sparse_s_qr_factorize(A, alt_values)
    @ccall libmkl_rt.mkl_sparse_s_qr_factorize(A::sparse_matrix_t,
                                               alt_values::Ptr{Float32})::sparse_status_t
end

function mkl_sparse_d_qr_solve(operation, A, alt_values, layout, columns, x, ldx, b, ldb)
    @ccall libmkl_rt.mkl_sparse_d_qr_solve(operation::sparse_operation_t,
                                           A::sparse_matrix_t, alt_values::Ptr{Float64},
                                           layout::sparse_layout_t, columns::BlasInt,
                                           x::Ptr{Float64}, ldx::BlasInt, b::Ptr{Float64},
                                           ldb::BlasInt)::sparse_status_t
end

function mkl_sparse_s_qr_solve(operation, A, alt_values, layout, columns, x, ldx, b, ldb)
    @ccall libmkl_rt.mkl_sparse_s_qr_solve(operation::sparse_operation_t,
                                           A::sparse_matrix_t, alt_values::Ptr{Float32},
                                           layout::sparse_layout_t, columns::BlasInt,
                                           x::Ptr{Float32}, ldx::BlasInt, b::Ptr{Float32},
                                           ldb::BlasInt)::sparse_status_t
end

function mkl_sparse_d_qr_qmult(operation, A, layout, columns, x, ldx, b, ldb)
    @ccall libmkl_rt.mkl_sparse_d_qr_qmult(operation::sparse_operation_t,
                                           A::sparse_matrix_t, layout::sparse_layout_t,
                                           columns::BlasInt, x::Ptr{Float64}, ldx::BlasInt,
                                           b::Ptr{Float64}, ldb::BlasInt)::sparse_status_t
end

function mkl_sparse_s_qr_qmult(operation, A, layout, columns, x, ldx, b, ldb)
    @ccall libmkl_rt.mkl_sparse_s_qr_qmult(operation::sparse_operation_t,
                                           A::sparse_matrix_t, layout::sparse_layout_t,
                                           columns::BlasInt, x::Ptr{Float32}, ldx::BlasInt,
                                           b::Ptr{Float32}, ldb::BlasInt)::sparse_status_t
end

function mkl_sparse_d_qr_rsolve(operation, A, layout, columns, x, ldx, b, ldb)
    @ccall libmkl_rt.mkl_sparse_d_qr_rsolve(operation::sparse_operation_t,
                                            A::sparse_matrix_t, layout::sparse_layout_t,
                                            columns::BlasInt, x::Ptr{Float64}, ldx::BlasInt,
                                            b::Ptr{Float64}, ldb::BlasInt)::sparse_status_t
end

function mkl_sparse_s_qr_rsolve(operation, A, layout, columns, x, ldx, b, ldb)
    @ccall libmkl_rt.mkl_sparse_s_qr_rsolve(operation::sparse_operation_t,
                                            A::sparse_matrix_t, layout::sparse_layout_t,
                                            columns::BlasInt, x::Ptr{Float32}, ldx::BlasInt,
                                            b::Ptr{Float32}, ldb::BlasInt)::sparse_status_t
end
