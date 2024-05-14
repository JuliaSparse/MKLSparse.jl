# The increments to the `__counter` variable is for testing purposes

function _check_transa(t::Char)
    if !(t in ('C', 'N', 'T'))
        error("transa: is '$t', must be 'N', 'T', or 'C'")
    end
end

mkl_size(t::Char, M::AbstractVecOrMat) = t == 'N' ? size(M) : reverse(size(M))


# Checks sizes for the multiplication C <- A * B
function _check_mat_mult_matvec(C, A, B, tA)
    _size(v::AbstractMatrix) = size(v)
    _size(v::AbstractVector) = (size(v,1), 1)
    _str(v::AbstractMatrix) = string("[", size(v,1), ", ", size(v,2), "]")
    _str(v::AbstractVector) = string("[", size(v,1), "]")
    mA, nA = mkl_size(tA, A)
    mB, nB = _size(B)
    mC, nC = _size(C)
    if nA != mB || mC != mA || nC != nB
        t = ""
        if tA == 'T'; t = ".\'"; end
        if tA == 'C'; t = "\'"; end
        str = string("arrays had inconsistent dimensions for C <- A", t, " * B: ", _str(C), " <- ", _str(A), t, " * ", _str(B))
        throw(DimensionMismatch(str))
    end
end

for (fname, T) in ((:mkl_scscmv, :Float32   ),
                   (:mkl_dcscmv, :Float64   ),
                   (:mkl_ccscmv, :ComplexF32),
                   (:mkl_zcscmv, :ComplexF64))
    @eval begin
        function cscmv!(transa::Char, α::$T, matdescra::String,
                        A::SparseMatrixCSC{$T, Int32}, x::StridedVector{$T},
                        β::$T, y::StridedVector{$T})
            _check_transa(transa)
            _check_mat_mult_matvec(y, A, x, transa)
            __counter[] += 1
            $fname(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y)
            return y
        end

        function cscmv!(transa::Char, α::$T, matdescra::String,
                        A::SparseMatrixCSC{$T, Int64}, x::StridedVector{$T},
                        β::$T, y::StridedVector{$T})
            _check_transa(transa)
            _check_mat_mult_matvec(y, A, x, transa)
            __counter[] += 1
            set_interface_layer(INTERFACE_ILP64)
            $fname(transa, A.m, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, β, y)
            set_interface_layer(INTERFACE_LP64)
            return y
        end
    end
end

for (fname, T) in ((:mkl_scscmm, :Float32   ),
                   (:mkl_dcscmm, :Float64   ),
                   (:mkl_ccscmm, :ComplexF32),
                   (:mkl_zcscmm, :ComplexF64))
    @eval begin
        function cscmm!(transa::Char, α::$T, matdescra::String,
                        A::SparseMatrixCSC{$T, Int32}, B::StridedMatrix{$T},
                        β::$T, C::StridedMatrix{$T})
            _check_transa(transa)
            _check_mat_mult_matvec(C, A, B, transa)
            mB, nB = size(B)
            mC, nC = size(C)
            __counter[] += 1
            $fname(transa, A.m, nC, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, β, C, mC)
            return C
        end

        function cscmm!(transa::Char, α::$T, matdescra::String,
                        A::SparseMatrixCSC{$T, Int64}, B::StridedMatrix{$T},
                        β::$T, C::StridedMatrix{$T})
            _check_transa(transa)
            _check_mat_mult_matvec(C, A, B, transa)
            mB, nB = size(B)
            mC, nC = size(C)
            __counter[] += 1
            set_interface_layer(INTERFACE_ILP64)
            $fname(transa, A.m, nC, A.n, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, β, C, mC)
            set_interface_layer(INTERFACE_LP64)
            return C
        end
    end
end

for (fname, T) in ((:mkl_scscsv, :Float32   ),
                   (:mkl_dcscsv, :Float64   ),
                   (:mkl_ccscsv, :ComplexF32),
                   (:mkl_zcscsv, :ComplexF64))
    @eval begin
        function cscsv!(transa::Char, α::$T, matdescra::String,
                        A::SparseMatrixCSC{$T, Int32}, x::StridedVector{$T},
                        y::StridedVector{$T})
            n = checksquare(A)
            _check_transa(transa)
            _check_mat_mult_matvec(y, A, x, transa)
            __counter[] += 1
            $fname(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y)
            return y
        end

        function cscsv!(transa::Char, α::$T, matdescra::String,
                        A::SparseMatrixCSC{$T, Int64}, x::StridedVector{$T},
                        y::StridedVector{$T})
            n = checksquare(A)
            _check_transa(transa)
            _check_mat_mult_matvec(y, A, x, transa)
            __counter[] += 1
            set_interface_layer(INTERFACE_ILP64)
            $fname(transa, A.m, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), x, y)
            set_interface_layer(INTERFACE_LP64)
            return y
        end
    end
end

for (fname, T) in ((:mkl_scscsm, :Float32   ),
                   (:mkl_dcscsm, :Float64   ),
                   (:mkl_ccscsm, :ComplexF32),
                   (:mkl_zcscsm, :ComplexF64))
    @eval begin
        function cscsm!(transa::Char, α::$T, matdescra::String,
                        A::SparseMatrixCSC{$T, Int32}, B::StridedMatrix{$T},
                        C::StridedMatrix{$T})
            mB, nB = size(B)
            mC, nC = size(C)
            n = checksquare(A)
            _check_transa(transa)
            _check_mat_mult_matvec(C, A, B, transa)
            __counter[] += 1
            $fname(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC)
            return C
        end

        function cscsm!(transa::Char, α::$T, matdescra::String,
                        A::SparseMatrixCSC{$T, Int64}, B::StridedMatrix{$T},
                        C::StridedMatrix{$T})
            mB, nB = size(B)
            mC, nC = size(C)
            n = checksquare(A)
            _check_transa(transa)
            _check_mat_mult_matvec(C, A, B, transa)
            __counter[] += 1
            set_interface_layer(INTERFACE_ILP64)
            $fname(transa, A.n, nC, α, matdescra, A.nzval, A.rowval, A.colptr, pointer(A.colptr, 2), B, mB, C, mC)
            set_interface_layer(INTERFACE_LP64)
            return C
        end
    end
end
