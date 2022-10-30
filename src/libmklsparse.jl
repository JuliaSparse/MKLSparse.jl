const MKL_LAYOUT = UInt32
const MKL_ROW_MAJOR = 101 % UInt32
const MKL_COL_MAJOR = 102 % UInt32

const MKL_TRANSPOSE = UInt32
const MKL_NOTRANS = 111 % UInt32
const MKL_TRANS = 112 % UInt32
const MKL_CONJTRANS = 113 % UInt32
const MKL_CONJ = 114 % UInt32

const MKL_UPLO = UInt32
const MKL_UPPER = 121 % UInt32
const MKL_LOWER = 122 % UInt32

const MKL_DIAG = UInt32
const MKL_NONUNIT = 131 % UInt32
const MKL_UNIT = 132 % UInt32

const MKL_SIDE = UInt32
const MKL_LEFT = 141 % UInt32
const MKL_RIGHT = 142 % UInt32

const MKL_COMPACT_PACK = UInt32
const MKL_COMPACT_SSE = 181 % UInt32
const MKL_COMPACT_AVX = 182 % UInt32
const MKL_COMPACT_AVX512 = 183 % UInt32

# typedef void ( * sgemm_jit_kernel_t ) ( void * , float * , float * , float * )
const sgemm_jit_kernel_t = Ptr{Cvoid}

# typedef void ( * dgemm_jit_kernel_t ) ( void * , double * , double * , double * )
const dgemm_jit_kernel_t = Ptr{Cvoid}

# typedef void ( * cgemm_jit_kernel_t ) ( void * , ComplexF32 * , ComplexF32 * , ComplexF32 * )
const cgemm_jit_kernel_t = Ptr{Cvoid}

# typedef void ( * zgemm_jit_kernel_t ) ( void * , ComplexF64 * , ComplexF64 * , ComplexF64 * )
const zgemm_jit_kernel_t = Ptr{Cvoid}

const mkl_jit_status_t = UInt32
const MKL_JIT_SUCCESS = 0 % UInt32
const MKL_NO_JIT = 1 % UInt32
const MKL_JIT_ERROR = 2 % UInt32

function mkl_scsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_scsrmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_scsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_scsrsv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_scsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_scsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_scsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_scsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_scsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_scsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, a::Ptr{Float32}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_scsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, a::Ptr{Float32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scscmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_scscmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_scscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_scscsv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_scoomv(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.mkl_scoomv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_scoosv(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_scoosv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_scoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_scoogemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_scoogemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_scoosymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_scoosymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_scootrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_scootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_scootrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_sdiamv(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.mkl_sdiamv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sdiasv(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_sdiasv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, lval::Ptr{BlasInt},
                                idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_sdiagemv(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_sdiagemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sdiasymv(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_sdiasymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sdiatrsv(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_sdiatrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{Float32}, lval::Ptr{BlasInt},
                                  idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_sskymv(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.mkl_sskymv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, x::Ptr{Float32}, beta::Ref{Float32},
                                y::Ptr{Float32})::Cvoid
end

function mkl_sskysv(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.mkl_sskysv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, pntr::Ptr{BlasInt},
                                x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrmv(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_sbsrmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrsv(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_sbsrsv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_sbsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_sbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_sbsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_sbsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_sbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_sbsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function mkl_sbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_sbsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function mkl_cspblas_sbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_sbsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function mkl_scsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_scsrmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, beta::Ref{Float32},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_scsrsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_scsrsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_scscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_scscmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, beta::Ref{Float32},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_scscsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_scscsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_scoomm(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_scoomm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, beta::Ref{Float32},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_scoosm(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_scoosm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_sdiamm(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_sdiamm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, beta::Ref{Float32},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_sdiasm(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_sdiasm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_sskysm(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_sskysm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_sskymm(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.mkl_sskymm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_sbsrmm(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_sbsrmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ptr{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{UInt8},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_sbsrsm(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_sbsrsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSRMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_SCSRMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SCSRSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_SCSRSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SCSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SCSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SCSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, a::Ptr{Float32}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, a::Ptr{Float32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCSCMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_SCSCMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SCSCSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_SCSCSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SCOOMV(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.MKL_SCOOMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SCOOSV(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_SCOOSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_SCOOGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCOOGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_SCOOSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCOOSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_SCOOTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SCOOTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, val::Ptr{Float32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SDIAMV(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.MKL_SDIAMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SDIASV(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_SDIASV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, lval::Ptr{BlasInt},
                                idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SDIAGEMV(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_SDIAGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SDIASYMV(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_SDIASYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float32},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SDIATRSV(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_SDIATRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{Float32}, lval::Ptr{BlasInt},
                                  idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_SSKYMV(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.MKL_SSKYMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, x::Ptr{Float32}, beta::Ref{Float32},
                                y::Ptr{Float32})::Cvoid
end

function MKL_SSKYSV(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.MKL_SSKYSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float32},
                                matdescra::Ptr{UInt8}, val::Ptr{Float32}, pntr::Ptr{BlasInt},
                                x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRMV(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_SBSRMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, beta::Ref{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRSV(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_SBSRSV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SBSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SBSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SBSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SBSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{Float32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function MKL_SBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_SBSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{Float32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                  y::Ptr{Float32})::Cvoid
end

function MKL_CSPBLAS_SBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_SBSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{Float32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function MKL_SCSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SCSRMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, beta::Ref{Float32},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSRSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_SCSRSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSCMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SCSCMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, beta::Ref{Float32},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSCSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_SCSCSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCOOMM(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SCOOMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, beta::Ref{Float32},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCOOSM(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_SCOOSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SDIAMM(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SDIAMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, beta::Ref{Float32},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SDIASM(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_SDIASM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_SSKYSM(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_SSKYSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SSKYMM(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.MKL_SSKYMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                pntr::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SBSRMM(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_SBSRMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ptr{BlasInt}, alpha::Ref{Float32}, matdescra::Ptr{UInt8},
                                val::Ptr{Float32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float32}, ldb::Ref{BlasInt},
                                beta::Ref{Float32}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SBSRSM(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_SBSRSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float32}, matdescra::Ptr{UInt8}, val::Ptr{Float32},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float32}, ldb::Ref{BlasInt}, c::Ptr{Float32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_dcsrmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dcsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_dcsrsv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dcsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dcsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dcsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, a::Ptr{Float64}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, a::Ptr{Float64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcscmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_dcscmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dcscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_dcscsv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcoomv(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.mkl_dcoomv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dcoosv(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_dcoosv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dcoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_dcoogemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcoogemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_dcoosymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcoosymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_dcootrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dcootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_dcootrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_ddiamv(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.mkl_ddiamv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_ddiasv(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_ddiasv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, lval::Ptr{BlasInt},
                                idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_ddiagemv(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_ddiagemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_ddiasymv(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_ddiasymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_ddiatrsv(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_ddiatrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{Float64}, lval::Ptr{BlasInt},
                                  idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_dskymv(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.mkl_dskymv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                pntr::Ptr{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function mkl_dskysv(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.mkl_dskysv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, pntr::Ptr{BlasInt},
                                x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrmv(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_dbsrmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrsv(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_dbsrsv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dbsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dbsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dbsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dbsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function mkl_dbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_dbsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function mkl_cspblas_dbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_dbsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function mkl_dcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_dcsrmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, beta::Ref{Float64},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcsrsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_dcsrsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_dcscmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, beta::Ref{Float64},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcscsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_dcscsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcoomm(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_dcoomm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, beta::Ref{Float64},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcoosm(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_dcoosm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_ddiamm(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_ddiamm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, beta::Ref{Float64},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ddiasm(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_ddiasm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_dskysm(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_dskysm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                pntr::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dskymm(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.mkl_dskymm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                pntr::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dbsrmm(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_dbsrmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ptr{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{UInt8},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dbsrsm(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_dbsrsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSRMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_DCSRMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DCSRSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_DCSRSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DCSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DCSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DCSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, a::Ptr{Float64}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, a::Ptr{Float64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCSCMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_DCSCMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DCSCSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_DCSCSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCOOMV(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.MKL_DCOOMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DCOOSV(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_DCOOSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_DCOOGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCOOGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_DCOOSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCOOSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_DCOOTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{Float64}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DCOOTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, val::Ptr{Float64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DDIAMV(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.MKL_DDIAMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DDIASV(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_DDIASV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, lval::Ptr{BlasInt},
                                idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DDIAGEMV(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_DDIAGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DDIASYMV(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_DDIASYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{Float64},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DDIATRSV(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_DDIATRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{Float64}, lval::Ptr{BlasInt},
                                  idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_DSKYMV(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.MKL_DSKYMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                pntr::Ptr{BlasInt}, x::Ptr{Float64}, beta::Ref{Float64},
                                y::Ptr{Float64})::Cvoid
end

function MKL_DSKYSV(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.MKL_DSKYSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{Float64},
                                matdescra::Ptr{UInt8}, val::Ptr{Float64}, pntr::Ptr{BlasInt},
                                x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRMV(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_DBSRMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float64}, beta::Ref{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRSV(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_DBSRSV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DBSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DBSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DBSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DBSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{Float64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function MKL_DBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_DBSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{Float64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                  y::Ptr{Float64})::Cvoid
end

function MKL_CSPBLAS_DBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_DBSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{Float64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function MKL_DCSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DCSRMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, beta::Ref{Float64},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSRSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_DCSRSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSCMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DCSCMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, beta::Ref{Float64},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSCSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_DCSCSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCOOMM(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DCOOMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, beta::Ref{Float64},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCOOSM(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_DCOOSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_DDIAMM(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DDIAMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, beta::Ref{Float64},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DDIASM(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_DDIASM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_DSKYSM(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_DSKYSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                pntr::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DSKYMM(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.MKL_DSKYMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                pntr::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DBSRMM(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_DBSRMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ptr{BlasInt}, alpha::Ref{Float64}, matdescra::Ptr{UInt8},
                                val::Ptr{Float64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{Float64}, ldb::Ref{BlasInt},
                                beta::Ref{Float64}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DBSRSM(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_DBSRSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{Float64}, matdescra::Ptr{UInt8}, val::Ptr{Float64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{Float64}, ldb::Ref{BlasInt}, c::Ptr{Float64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_ccsrmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_ccsrsv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_ccsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_ccsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_ccsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccscmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_ccscmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_ccscsv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccoomv(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.mkl_ccoomv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccoosv(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_ccoosv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_ccoogemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccoogemv(transa::Ref{UInt8}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_ccoosymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccoosymv(uplo::Ref{UInt8}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_ccootrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_ccootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_ccootrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiamv(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.mkl_cdiamv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiasv(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_cdiasv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32}, lval::Ptr{BlasInt},
                                idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiagemv(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_cdiagemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiasymv(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_cdiasymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cdiatrsv(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_cdiatrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF32}, lval::Ptr{BlasInt},
                                  idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cskymv(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.mkl_cskymv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, beta::Ref{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_cskysv(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.mkl_cskysv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrmv(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_cbsrmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrsv(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_cbsrsv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cbsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{ComplexF32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_cbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_cbsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cbsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{ComplexF32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_cbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_cbsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function mkl_cbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cbsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function mkl_cspblas_cbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_cbsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{ComplexF32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function mkl_ccsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_ccsrmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccsrsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_ccsrsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_ccscmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccscsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_ccscsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccoomm(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_ccoomm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, b::Ptr{ComplexF32},
                                ldb::Ref{BlasInt}, beta::Ref{ComplexF32},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccoosm(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_ccoosm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, b::Ptr{ComplexF32},
                                ldb::Ref{BlasInt}, c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_cdiamm(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_cdiamm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_cdiasm(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_cdiasm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_cskysm(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_cskysm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_cskymm(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.mkl_cskymm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_cbsrmm(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_cbsrmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ptr{BlasInt}, alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_cbsrsm(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_cbsrsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSRMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_CCSRMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_CCSRSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CCSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CCSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CCSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSCMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_CCSCMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSCSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_CCSCSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32}, indx::Ptr{BlasInt},
                                pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOMV(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.MKL_CCOOMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOSV(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CCOOSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CCOOGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCOOGEMV(transa::Ref{UInt8}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CCOOSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCOOSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CCOOTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CCOOTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIAMV(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.MKL_CDIAMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIASV(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_CDIASV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32}, lval::Ptr{BlasInt},
                                idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIAGEMV(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_CDIAGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIASYMV(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_CDIASYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF32},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CDIATRSV(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_CDIATRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF32}, lval::Ptr{BlasInt},
                                  idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSKYMV(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.MKL_CSKYMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, beta::Ref{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSKYSV(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.MKL_CSKYSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF32},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRMV(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_CBSRMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                beta::Ref{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRSV(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_CBSRSV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CBSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{ComplexF32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CBSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CBSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{ComplexF32}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CBSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF32}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                          y::Ptr{ComplexF32})::Cvoid
end

function MKL_CBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CBSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{ComplexF32},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF32},
                                  y::Ptr{ComplexF32})::Cvoid
end

function MKL_CSPBLAS_CBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_CBSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{ComplexF32},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid
end

function MKL_CCSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CCSRMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSRSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_CCSRSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSCMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CCSCMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSCSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_CCSCSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCOOMM(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CCOOMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, b::Ptr{ComplexF32},
                                ldb::Ref{BlasInt}, beta::Ref{ComplexF32},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCOOSM(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_CCOOSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, b::Ptr{ComplexF32},
                                ldb::Ref{BlasInt}, c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CDIAMM(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CDIAMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CDIASM(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_CDIASM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_CSKYSM(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_CSKYSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CSKYMM(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.MKL_CSKYMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CBSRMM(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_CBSRMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ptr{BlasInt}, alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF32}, c::Ptr{ComplexF32},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_CBSRSM(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_CBSRSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF32}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF32}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF32}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_zcsrmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_zcsrsv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zcsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcsrgemv(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zcsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcsrsymv(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zcsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcsrtrsv(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcscmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_zcscmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcscsv(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_zcscsv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcoomv(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.mkl_zcoomv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcoosv(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_zcoosv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_zcoogemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcoogemv(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcoogemv(transa::Ref{UInt8}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_zcoosymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcoosymv(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcoosymv(uplo::Ref{UInt8}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_zcootrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zcootrsv(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.mkl_cspblas_zcootrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiamv(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.mkl_zdiamv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiasv(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_zdiasv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiagemv(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_zdiagemv(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiasymv(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_zdiasymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zdiatrsv(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.mkl_zdiatrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF64}, lval::Ptr{BlasInt},
                                  idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_zskymv(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.mkl_zskymv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, beta::Ref{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function mkl_zskysv(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.mkl_zskysv(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                pntr::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrmv(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.mkl_zbsrmv(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrsv(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.mkl_zbsrsv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zbsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{ComplexF64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zbsrgemv(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zbsrgemv(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zbsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{ComplexF64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zbsrsymv(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zbsrsymv(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_zbsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function mkl_cspblas_zbsrtrsv(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.mkl_cspblas_zbsrtrsv(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function mkl_zcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zcsrmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcsrsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_zcsrsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcscmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zcscmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcscsm(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_zcscsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcoomm(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zcoomm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, b::Ptr{ComplexF64},
                                ldb::Ref{BlasInt}, beta::Ref{ComplexF64},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcoosm(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_zcoosm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, b::Ptr{ComplexF64},
                                ldb::Ref{BlasInt}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zdiamm(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zdiamm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zdiasm(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_zdiasm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zskysm(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.mkl_zskysm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_zskymm(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.mkl_zskymm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zbsrmm(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.mkl_zbsrmm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ptr{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function mkl_zbsrsm(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.mkl_zbsrsm(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_ZCSRMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_ZCSRSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZCSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCSRGEMV(transa, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZCSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCSRSYMV(uplo, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZCSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                  ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCSRTRSV(uplo, transa, diag, m, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, a::Ptr{ComplexF64},
                                          ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSCMV(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_ZCSCMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSCSV(transa, m, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_ZCSCSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOMV(transa, m, k, alpha, matdescra, val, rowind, colind, nnz, x, beta, y)
    @ccall libmkl_rt.MKL_ZCOOMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOSV(transa, m, alpha, matdescra, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_ZCOOSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_ZCOOGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCOOGEMV(transa, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCOOGEMV(transa::Ref{UInt8}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_ZCOOSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  rowind::Ptr{BlasInt}, colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCOOSYMV(uplo, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCOOSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt},
                                          val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                          colind::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                          x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_ZCOOTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                  colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZCOOTRSV(uplo, transa, diag, m, val, rowind, colind, nnz, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZCOOTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                          rowind::Ptr{BlasInt}, colind::Ptr{BlasInt},
                                          nnz::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIAMV(transa, m, k, alpha, matdescra, val, lval, idiag, ndiag, x, beta, y)
    @ccall libmkl_rt.MKL_ZDIAMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIASV(transa, m, alpha, matdescra, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_ZDIASV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIAGEMV(transa, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_ZDIAGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIASYMV(uplo, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_ZDIASYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, val::Ptr{ComplexF64},
                                  lval::Ptr{BlasInt}, idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZDIATRSV(uplo, transa, diag, m, val, lval, idiag, ndiag, x, y)
    @ccall libmkl_rt.MKL_ZDIATRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, val::Ptr{ComplexF64}, lval::Ptr{BlasInt},
                                  idiag::Ptr{BlasInt}, ndiag::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZSKYMV(transa, m, k, alpha, matdescra, val, pntr, x, beta, y)
    @ccall libmkl_rt.MKL_ZSKYMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                x::Ptr{ComplexF64}, beta::Ref{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZSKYSV(transa, m, alpha, matdescra, val, pntr, x, y)
    @ccall libmkl_rt.MKL_ZSKYSV(transa::Ref{UInt8}, m::Ref{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                pntr::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRMV(transa, m, k, lb, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    @ccall libmkl_rt.MKL_ZBSRMV(transa::Ref{UInt8}, m::Ref{BlasInt}, k::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                beta::Ref{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRSV(transa, m, lb, alpha, matdescra, val, indx, pntrb, pntre, x, y)
    @ccall libmkl_rt.MKL_ZBSRSV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZBSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{ComplexF64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZBSRGEMV(transa, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZBSRGEMV(transa::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZBSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                  a::Ptr{ComplexF64}, ia::Ptr{BlasInt}, ja::Ptr{BlasInt},
                                  x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZBSRSYMV(uplo, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZBSRSYMV(uplo::Ref{UInt8}, m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_ZBSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                  m::Ref{BlasInt}, lb::Ptr{BlasInt}, a::Ptr{ComplexF64},
                                  ia::Ptr{BlasInt}, ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                  y::Ptr{ComplexF64})::Cvoid
end

function MKL_CSPBLAS_ZBSRTRSV(uplo, transa, diag, m, lb, a, ia, ja, x, y)
    @ccall libmkl_rt.MKL_CSPBLAS_ZBSRTRSV(uplo::Ref{UInt8}, transa::Ref{UInt8}, diag::Ref{UInt8},
                                          m::Ref{BlasInt}, lb::Ptr{BlasInt},
                                          a::Ptr{ComplexF64}, ia::Ptr{BlasInt},
                                          ja::Ptr{BlasInt}, x::Ptr{ComplexF64},
                                          y::Ptr{ComplexF64})::Cvoid
end

function MKL_ZCSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZCSRMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_ZCSRSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSCMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZCSCMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSCSM(transa, m, n, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_ZCSCSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCOOMM(transa, m, n, k, alpha, matdescra, val, rowind, colind, nnz, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZCOOMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, b::Ptr{ComplexF64},
                                ldb::Ref{BlasInt}, beta::Ref{ComplexF64},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCOOSM(transa, m, n, alpha, matdescra, val, rowind, colind, nnz, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_ZCOOSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, rowind::Ptr{BlasInt},
                                colind::Ptr{BlasInt}, nnz::Ptr{BlasInt}, b::Ptr{ComplexF64},
                                ldb::Ref{BlasInt}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZDIAMM(transa, m, n, k, alpha, matdescra, val, lval, idiag, ndiag, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZDIAMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZDIASM(transa, m, n, alpha, matdescra, val, lval, idiag, ndiag, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_ZDIASM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, lval::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                ndiag::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZSKYSM(transa, m, n, alpha, matdescra, val, pntr, b, ldb, c, ldc)
    @ccall libmkl_rt.MKL_ZSKYSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZSKYMM(transa, m, n, k, alpha, matdescra, val, pntr, b, ldb, beta, c, ldc)
    @ccall libmkl_rt.MKL_ZSKYMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, pntr::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZBSRMM(transa, m, n, k, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb,
                    beta, c, ldc)
    @ccall libmkl_rt.MKL_ZBSRMM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                lb::Ptr{BlasInt}, alpha::Ref{ComplexF64},
                                matdescra::Ptr{UInt8}, val::Ptr{ComplexF64},
                                indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt}, pntre::Ptr{BlasInt},
                                b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                beta::Ref{ComplexF64}, c::Ptr{ComplexF64},
                                ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZBSRSM(transa, m, n, lb, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, c,
                    ldc)
    @ccall libmkl_rt.MKL_ZBSRSM(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt}, lb::Ptr{BlasInt},
                                alpha::Ref{ComplexF64}, matdescra::Ptr{UInt8},
                                val::Ptr{ComplexF64}, indx::Ptr{BlasInt}, pntrb::Ptr{BlasInt},
                                pntre::Ptr{BlasInt}, b::Ptr{ComplexF64}, ldb::Ref{BlasInt},
                                c::Ptr{ComplexF64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcsrbsr(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.mkl_dcsrbsr(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ptr{BlasInt}, Acsr::Ptr{Float64}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{Float64}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_dcsrcoo(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.mkl_dcsrcoo(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                 Acoo::Ptr{Float64}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function mkl_ddnscsr(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.mkl_ddnscsr(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{Float64}, lda::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ::Ptr{BlasInt}, AI::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_dcsrcsc(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.mkl_dcsrcsc(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{Float64},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_dcsrdia(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.mkl_dcsrdia(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{Float64},
                                 ndiag::Ptr{BlasInt}, distance::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                 Acsr_rem::Ref{Float64}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_dcsrsky(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.mkl_dcsrsky(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{Float64},
                                 pointers::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_scsrbsr(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.mkl_scsrbsr(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ptr{BlasInt}, Acsr::Ptr{Float32}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{Float32}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_scsrcoo(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.mkl_scsrcoo(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                 Acoo::Ptr{Float32}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function mkl_sdnscsr(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.mkl_sdnscsr(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{Float32}, lda::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ::Ptr{BlasInt}, AI::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_scsrcsc(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.mkl_scsrcsc(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{Float32},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_scsrdia(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.mkl_scsrdia(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{Float32},
                                 ndiag::Ptr{BlasInt}, distance::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                 Acsr_rem::Ref{Float32}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_scsrsky(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.mkl_scsrsky(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{Float32},
                                 pointers::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_ccsrbsr(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.mkl_ccsrbsr(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ptr{BlasInt}, Acsr::Ptr{ComplexF32}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{ComplexF32}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_ccsrcoo(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.mkl_ccsrcoo(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                 Acoo::Ptr{ComplexF32}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function mkl_cdnscsr(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.mkl_cdnscsr(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{ComplexF32}, lda::Ref{BlasInt},
                                 Acsr::Ptr{ComplexF32}, AJ::Ptr{BlasInt}, AI::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function mkl_ccsrcsc(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.mkl_ccsrcsc(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{ComplexF32},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_ccsrdia(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.mkl_ccsrdia(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{ComplexF32},
                                 ndiag::Ptr{BlasInt}, distance::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                 Acsr_rem::Ref{ComplexF32}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_ccsrsky(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.mkl_ccsrsky(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{ComplexF32},
                                 pointers::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_zcsrbsr(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.mkl_zcsrbsr(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ptr{BlasInt}, Acsr::Ptr{ComplexF64}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{ComplexF64}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_zcsrcoo(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.mkl_zcsrcoo(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                 Acoo::Ptr{ComplexF64}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function mkl_zdnscsr(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.mkl_zdnscsr(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{ComplexF64}, lda::Ref{BlasInt},
                                 Acsr::Ptr{ComplexF64}, AJ::Ptr{BlasInt}, AI::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function mkl_zcsrcsc(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.mkl_zcsrcsc(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{ComplexF64},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_zcsrdia(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.mkl_zcsrdia(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{ComplexF64},
                                 ndiag::Ptr{BlasInt}, distance::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                 Acsr_rem::Ref{ComplexF64}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_zcsrsky(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.mkl_zcsrsky(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{ComplexF64},
                                 pointers::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_DCSRBSR(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.MKL_DCSRBSR(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ptr{BlasInt}, Acsr::Ptr{Float64}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{Float64}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_DCSRCOO(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.MKL_DCSRCOO(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                 Acoo::Ptr{Float64}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function MKL_DDNSCSR(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.MKL_DDNSCSR(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{Float64}, lda::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ::Ptr{BlasInt}, AI::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_DCSRCSC(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.MKL_DCSRCSC(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{Float64},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_DCSRDIA(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.MKL_DCSRDIA(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{Float64},
                                 ndiag::Ptr{BlasInt}, distance::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                 Acsr_rem::Ref{Float64}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_DCSRSKY(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.MKL_DCSRSKY(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{Float64},
                                 pointers::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_SCSRBSR(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.MKL_SCSRBSR(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ptr{BlasInt}, Acsr::Ptr{Float32}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{Float32}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_SCSRCOO(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.MKL_SCSRCOO(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                 Acoo::Ptr{Float32}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function MKL_SDNSCSR(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.MKL_SDNSCSR(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{Float32}, lda::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ::Ptr{BlasInt}, AI::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_SCSRCSC(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.MKL_SCSRCSC(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{Float32},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_SCSRDIA(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.MKL_SCSRDIA(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{Float32},
                                 ndiag::Ptr{BlasInt}, distance::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                 Acsr_rem::Ref{Float32}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_SCSRSKY(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.MKL_SCSRSKY(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{Float32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{Float32},
                                 pointers::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_CCSRBSR(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.MKL_CCSRBSR(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ptr{BlasInt}, Acsr::Ptr{ComplexF32}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{ComplexF32}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_CCSRCOO(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.MKL_CCSRCOO(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                 Acoo::Ptr{ComplexF32}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function MKL_CDNSCSR(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.MKL_CDNSCSR(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{ComplexF32}, lda::Ref{BlasInt},
                                 Acsr::Ptr{ComplexF32}, AJ::Ptr{BlasInt}, AI::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function MKL_CCSRCSC(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.MKL_CCSRCSC(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{ComplexF32},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_CCSRDIA(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.MKL_CCSRDIA(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{ComplexF32},
                                 ndiag::Ptr{BlasInt}, distance::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                 Acsr_rem::Ref{ComplexF32}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_CCSRSKY(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.MKL_CCSRSKY(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF32},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{ComplexF32},
                                 pointers::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_ZCSRBSR(job, m, mblk, ldAbsr, Acsr, AJ, AI, Absr, AJB, AIB, info)
    @ccall libmkl_rt.MKL_ZCSRBSR(job::Ref{BlasInt}, m::Ref{BlasInt}, mblk::Ref{BlasInt},
                                 ldAbsr::Ptr{BlasInt}, Acsr::Ptr{ComplexF64}, AJ::Ptr{BlasInt},
                                 AI::Ptr{BlasInt}, Absr::Ptr{ComplexF64}, AJB::Ptr{BlasInt},
                                 AIB::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_ZCSRCOO(job, n, Acsr, AJR, AIR, nnz, Acoo, ir, jc, info)
    @ccall libmkl_rt.MKL_ZCSRCOO(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJR::Ptr{BlasInt}, AIR::Ptr{BlasInt}, nnz::Ptr{BlasInt},
                                 Acoo::Ptr{ComplexF64}, ir::Ptr{BlasInt}, jc::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function MKL_ZDNSCSR(job, m, n, Adns, lda, Acsr, AJ, AI, info)
    @ccall libmkl_rt.MKL_ZDNSCSR(job::Ref{BlasInt}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                 Adns::Ptr{ComplexF64}, lda::Ref{BlasInt},
                                 Acsr::Ptr{ComplexF64}, AJ::Ptr{BlasInt}, AI::Ptr{BlasInt},
                                 info::Ptr{BlasInt})::Cvoid
end

function MKL_ZCSRCSC(job, n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info)
    @ccall libmkl_rt.MKL_ZCSRCSC(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Acsc::Ptr{ComplexF64},
                                 AJ1::Ptr{BlasInt}, AI1::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_ZCSRDIA(job, n, Acsr, AJ0, AI0, Adia, ndiag, distance, idiag, Acsr_rem,
                     AJ0_rem, AI0_rem, info)
    @ccall libmkl_rt.MKL_ZCSRDIA(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Adia::Ptr{ComplexF64},
                                 ndiag::Ptr{BlasInt}, distance::Ptr{BlasInt}, idiag::Ptr{BlasInt},
                                 Acsr_rem::Ref{ComplexF64}, AJ0_rem::Ref{BlasInt},
                                 AI0_rem::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function MKL_ZCSRSKY(job, n, Acsr, AJ0, AI0, Asky, pointers, info)
    @ccall libmkl_rt.MKL_ZCSRSKY(job::Ref{BlasInt}, n::Ref{BlasInt}, Acsr::Ptr{ComplexF64},
                                 AJ0::Ptr{BlasInt}, AI0::Ptr{BlasInt}, Asky::Ptr{ComplexF64},
                                 pointers::Ptr{BlasInt}, info::Ptr{BlasInt})::Cvoid
end

function mkl_dcsrmultcsr(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.mkl_dcsrmultcsr(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{Float64}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{Float64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{Float64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function mkl_dcsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.mkl_dcsrmultd(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{Float64}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{Float64}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_dcsradd(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.mkl_dcsradd(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{Float64}, ja::Ptr{BlasInt},
                                 ia::Ptr{BlasInt}, beta::Ref{Float64}, b::Ptr{Float64},
                                 jb::Ptr{BlasInt}, ib::Ptr{BlasInt}, c::Ptr{Float64},
                                 jc::Ptr{BlasInt}, ic::Ptr{BlasInt}, nnzmax::Ptr{BlasInt},
                                 ierr::Ptr{BlasInt})::Cvoid
end

function mkl_scsrmultcsr(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.mkl_scsrmultcsr(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{Float32}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{Float32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{Float32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function mkl_scsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.mkl_scsrmultd(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{Float32}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{Float32}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function mkl_scsradd(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.mkl_scsradd(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{Float32}, ja::Ptr{BlasInt},
                                 ia::Ptr{BlasInt}, beta::Ref{Float32}, b::Ptr{Float32},
                                 jb::Ptr{BlasInt}, ib::Ptr{BlasInt}, c::Ptr{Float32},
                                 jc::Ptr{BlasInt}, ic::Ptr{BlasInt}, nnzmax::Ptr{BlasInt},
                                 ierr::Ptr{BlasInt})::Cvoid
end

function mkl_ccsrmultcsr(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.mkl_ccsrmultcsr(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{ComplexF32}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{ComplexF32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{ComplexF32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function mkl_ccsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.mkl_ccsrmultd(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{ComplexF32}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{ComplexF32}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{ComplexF32},
                                   ldc::Ref{BlasInt})::Cvoid
end

function mkl_ccsradd(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.mkl_ccsradd(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{ComplexF32},
                                 ja::Ptr{BlasInt}, ia::Ptr{BlasInt}, beta::Ref{ComplexF32},
                                 b::Ptr{ComplexF32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                 c::Ptr{ComplexF32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                 nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function mkl_zcsrmultcsr(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.mkl_zcsrmultcsr(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{ComplexF64}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{ComplexF64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{ComplexF64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function mkl_zcsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.mkl_zcsrmultd(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{ComplexF64}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{ComplexF64}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{ComplexF64},
                                   ldc::Ref{BlasInt})::Cvoid
end

function mkl_zcsradd(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.mkl_zcsradd(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{ComplexF64},
                                 ja::Ptr{BlasInt}, ia::Ptr{BlasInt}, beta::Ref{ComplexF64},
                                 b::Ptr{ComplexF64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                 c::Ptr{ComplexF64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                 nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function MKL_DCSRMULTCSR(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.MKL_DCSRMULTCSR(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{Float64}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{Float64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{Float64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function MKL_DCSRMULTD(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.MKL_DCSRMULTD(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{Float64}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{Float64}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{Float64}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_DCSRADD(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.MKL_DCSRADD(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{Float64}, ja::Ptr{BlasInt},
                                 ia::Ptr{BlasInt}, beta::Ref{Float64}, b::Ptr{Float64},
                                 jb::Ptr{BlasInt}, ib::Ptr{BlasInt}, c::Ptr{Float64},
                                 jc::Ptr{BlasInt}, ic::Ptr{BlasInt}, nnzmax::Ptr{BlasInt},
                                 ierr::Ptr{BlasInt})::Cvoid
end

function MKL_SCSRMULTCSR(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.MKL_SCSRMULTCSR(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{Float32}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{Float32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{Float32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function MKL_SCSRMULTD(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.MKL_SCSRMULTD(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{Float32}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{Float32}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{Float32}, ldc::Ref{BlasInt})::Cvoid
end

function MKL_SCSRADD(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.MKL_SCSRADD(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{Float32}, ja::Ptr{BlasInt},
                                 ia::Ptr{BlasInt}, beta::Ref{Float32}, b::Ptr{Float32},
                                 jb::Ptr{BlasInt}, ib::Ptr{BlasInt}, c::Ptr{Float32},
                                 jc::Ptr{BlasInt}, ic::Ptr{BlasInt}, nnzmax::Ptr{BlasInt},
                                 ierr::Ptr{BlasInt})::Cvoid
end

function MKL_CCSRMULTCSR(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.MKL_CCSRMULTCSR(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{ComplexF32}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{ComplexF32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{ComplexF32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function MKL_CCSRMULTD(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.MKL_CCSRMULTD(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{ComplexF32}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{ComplexF32}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{ComplexF32},
                                   ldc::Ref{BlasInt})::Cvoid
end

function MKL_CCSRADD(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.MKL_CCSRADD(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{ComplexF32},
                                 ja::Ptr{BlasInt}, ia::Ptr{BlasInt}, beta::Ref{ComplexF32},
                                 b::Ptr{ComplexF32}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                 c::Ptr{ComplexF32}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                 nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function MKL_ZCSRMULTCSR(transa, job, sort, m, n, k, a, ja, ia, b, jb, ib, c, jc, ic,
                         nnzmax, ierr)
    @ccall libmkl_rt.MKL_ZCSRMULTCSR(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                     m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                     a::Ptr{ComplexF64}, ja::Ptr{BlasInt}, ia::Ptr{BlasInt},
                                     b::Ptr{ComplexF64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                     c::Ptr{ComplexF64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                     nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

function MKL_ZCSRMULTD(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc)
    @ccall libmkl_rt.MKL_ZCSRMULTD(transa::Ref{UInt8}, m::Ref{BlasInt}, n::Ref{BlasInt},
                                   k::Ref{BlasInt}, a::Ptr{ComplexF64}, ja::Ptr{BlasInt},
                                   ia::Ptr{BlasInt}, b::Ptr{ComplexF64}, jb::Ptr{BlasInt},
                                   ib::Ptr{BlasInt}, c::Ptr{ComplexF64},
                                   ldc::Ref{BlasInt})::Cvoid
end

function MKL_ZCSRADD(transa, job, sort, m, n, a, ja, ia, beta, b, jb, ib, c, jc, ic, nnzmax,
                     ierr)
    @ccall libmkl_rt.MKL_ZCSRADD(transa::Ref{UInt8}, job::Ref{BlasInt}, sort::Ptr{BlasInt},
                                 m::Ref{BlasInt}, n::Ref{BlasInt}, a::Ptr{ComplexF64},
                                 ja::Ptr{BlasInt}, ia::Ptr{BlasInt}, beta::Ref{ComplexF64},
                                 b::Ptr{ComplexF64}, jb::Ptr{BlasInt}, ib::Ptr{BlasInt},
                                 c::Ptr{ComplexF64}, jc::Ptr{BlasInt}, ic::Ptr{BlasInt},
                                 nnzmax::Ptr{BlasInt}, ierr::Ptr{BlasInt})::Cvoid
end

const sparse_status_t = UInt32
const SPARSE_STATUS_SUCCESS = 0 % UInt32
const SPARSE_STATUS_NOT_INITIALIZED = 1 % UInt32
const SPARSE_STATUS_ALLOC_FAILED = 2 % UInt32
const SPARSE_STATUS_INVALID_VALUE = 3 % UInt32
const SPARSE_STATUS_EXECUTION_FAILED = 4 % UInt32
const SPARSE_STATUS_INTERNAL_ERROR = 5 % UInt32
const SPARSE_STATUS_NOT_SUPPORTED = 6 % UInt32

const sparse_operation_t = UInt32
const SPARSE_OPERATION_NON_TRANSPOSE = 10 % UInt32
const SPARSE_OPERATION_TRANSPOSE = 11 % UInt32
const SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12 % UInt32

const sparse_matrix_type_t = UInt32
const SPARSE_MATRIX_TYPE_GENERAL = 20 % UInt32
const SPARSE_MATRIX_TYPE_SYMMETRIC = 21 % UInt32
const SPARSE_MATRIX_TYPE_HERMITIAN = 22 % UInt32
const SPARSE_MATRIX_TYPE_TRIANGULAR = 23 % UInt32
const SPARSE_MATRIX_TYPE_DIAGONAL = 24 % UInt32
const SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 25 % UInt32
const SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 26 % UInt32

const sparse_index_base_t = UInt32
const SPARSE_INDEX_BASE_ZERO = 0 % UInt32
const SPARSE_INDEX_BASE_ONE = 1 % UInt32

const sparse_fill_mode_t = UInt32
const SPARSE_FILL_MODE_LOWER = 40 % UInt32
const SPARSE_FILL_MODE_UPPER = 41 % UInt32
const SPARSE_FILL_MODE_FULL = 42 % UInt32

const sparse_diag_type_t = UInt32
const SPARSE_DIAG_NON_UNIT = 50 % UInt32
const SPARSE_DIAG_UNIT = 51 % UInt32

const sparse_layout_t = UInt32
const SPARSE_LAYOUT_ROW_MAJOR = 101 % UInt32
const SPARSE_LAYOUT_COLUMN_MAJOR = 102 % UInt32

const verbose_mode_t = UInt32
const SPARSE_VERBOSE_OFF = 70 % UInt32
const SPARSE_VERBOSE_BASIC = 71 % UInt32
const SPARSE_VERBOSE_EXTENDED = 72 % UInt32

const sparse_memory_usage_t = UInt32
const SPARSE_MEMORY_NONE = 80 % UInt32
const SPARSE_MEMORY_AGGRESSIVE = 81 % UInt32

const sparse_request_t = UInt32
const SPARSE_STAGE_FULL_MULT = 90 % UInt32
const SPARSE_STAGE_NNZ_COUNT = 91 % UInt32
const SPARSE_STAGE_FINALIZE_MULT = 92 % UInt32
const SPARSE_STAGE_FULL_MULT_NO_VAL = 93 % UInt32
const SPARSE_STAGE_FINALIZE_MULT_NO_VAL = 94 % UInt32

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

function mkl_sparse_copy(source, descr, dest)
    @ccall libmkl_rt.mkl_sparse_copy(source::sparse_matrix_t, descr::matrix_descr,
                                     dest::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_destroy(A)
    @ccall libmkl_rt.mkl_sparse_destroy(A::sparse_matrix_t)::sparse_status_t
end

function mkl_sparse_get_error_info(A, info)
    @ccall libmkl_rt.mkl_sparse_get_error_info(A::sparse_matrix_t,
                                               info::Ptr{BlasInt})::sparse_status_t
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

function mkl_sparse_set_verbose_mode(verbose)
    @ccall libmkl_rt.mkl_sparse_set_verbose_mode(verbose::verbose_mode_t)::sparse_status_t
end

function mkl_sparse_set_mv_hint(A, operation, descr, expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_mv_hint(A::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            descr::matrix_descr,
                                            expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_dotmv_hint(A, operation, descr, expectedCalls)
    @ccall libmkl_rt.mkl_sparse_set_dotmv_hint(A::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               descr::matrix_descr,
                                               expectedCalls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_mm_hint(A, operation, descr, layout, dense_matrix_size,
                                expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_mm_hint(A::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            descr::matrix_descr, layout::sparse_layout_t,
                                            dense_matrix_size::BlasInt,
                                            expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_sv_hint(A, operation, descr, expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_sv_hint(A::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            descr::matrix_descr,
                                            expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_sm_hint(A, operation, descr, layout, dense_matrix_size,
                                expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_sm_hint(A::sparse_matrix_t,
                                            operation::sparse_operation_t,
                                            descr::matrix_descr, layout::sparse_layout_t,
                                            dense_matrix_size::BlasInt,
                                            expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_symgs_hint(A, operation, descr, expected_calls)
    @ccall libmkl_rt.mkl_sparse_set_symgs_hint(A::sparse_matrix_t,
                                               operation::sparse_operation_t,
                                               descr::matrix_descr,
                                               expected_calls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_lu_smoother_hint(A, operation, descr, expectedCalls)
    @ccall libmkl_rt.mkl_sparse_set_lu_smoother_hint(A::sparse_matrix_t,
                                                     operation::sparse_operation_t,
                                                     descr::matrix_descr,
                                                     expectedCalls::BlasInt)::sparse_status_t
end

function mkl_sparse_set_memory_hint(A, policy)
    @ccall libmkl_rt.mkl_sparse_set_memory_hint(A::sparse_matrix_t,
                                                policy::sparse_memory_usage_t)::sparse_status_t
end

function mkl_sparse_optimize(A)
    @ccall libmkl_rt.mkl_sparse_optimize(A::sparse_matrix_t)::sparse_status_t
end

function mkl_sparse_order(A)
    @ccall libmkl_rt.mkl_sparse_order(A::sparse_matrix_t)::sparse_status_t
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

function mkl_sparse_spmm(operation, A, B, C)
    @ccall libmkl_rt.mkl_sparse_spmm(operation::sparse_operation_t, A::sparse_matrix_t,
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

function mkl_sparse_syrk(operation, A, C)
    @ccall libmkl_rt.mkl_sparse_syrk(operation::sparse_operation_t, A::sparse_matrix_t,
                                     C::Ptr{sparse_matrix_t})::sparse_status_t
end

function mkl_sparse_sypr(transA, A, B, descrB, C, request)
    @ccall libmkl_rt.mkl_sparse_sypr(transA::sparse_operation_t, A::sparse_matrix_t,
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

const MKL_INT64 = Clonglong

const MKL_UINT64 = Culonglong

const MKL_INT = BlasInt

const MKL_UINT = Cuint

# Skipping MacroDefinition: MKL_LONG long int

const MKL_UINT8 = Cuchar

const MKL_INT8 = Cchar

const MKL_INT16 = Cshort

const MKL_BF16 = Cushort

const MKL_INT32 = BlasInt

const MKL_F16 = Cushort

const MKL_DOMAIN_ALL = 0

const MKL_DOMAIN_BLAS = 1

const MKL_DOMAIN_FFT = 2

const MKL_DOMAIN_VML = 3

const MKL_DOMAIN_PARDISO = 4

const MKL_DOMAIN_LAPACK = 5

const MKL_CBWR_BRANCH = 1

const MKL_CBWR_ALL = ~0

const MKL_CBWR_STRICT = 0x00010000

const MKL_CBWR_OFF = 0

const MKL_CBWR_UNSET_ALL = MKL_CBWR_OFF

const MKL_CBWR_BRANCH_OFF = 1

const MKL_CBWR_AUTO = 2

const MKL_CBWR_COMPATIBLE = 3

const MKL_CBWR_SSE2 = 4

const MKL_CBWR_SSSE3 = 6

const MKL_CBWR_SSE4_1 = 7

const MKL_CBWR_SSE4_2 = 8

const MKL_CBWR_AVX = 9

const MKL_CBWR_AVX2 = 10

const MKL_CBWR_AVX512_MIC = 11

const MKL_CBWR_AVX512 = 12

const MKL_CBWR_AVX512_MIC_E1 = 13

const MKL_CBWR_AVX512_E1 = 14

const MKL_CBWR_SUCCESS = 0

const MKL_CBWR_ERR_INVALID_SETTINGS = -1

const MKL_CBWR_ERR_INVALID_INPUT = -2

const MKL_CBWR_ERR_UNSUPPORTED_BRANCH = -3

const MKL_CBWR_ERR_UNKNOWN_BRANCH = -4

const MKL_CBWR_ERR_MODE_CHANGE_FAILURE = -8

const MKL_CBWR_SSE3 = 5

# Skipping MacroDefinition: MKL_DEPRECATED __attribute__ ( ( deprecated ) )
