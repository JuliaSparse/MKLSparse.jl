MKLFloats = Union{Float32,Float64,ComplexF32,ComplexF64}

"""
    MKLcsc{T<:MKLFloats}

The opaque struct `sparse_matrix` from the MKL library in compressed sparse column (CSC) format
"""
mutable struct MKLcsc{T<:MKLFloats}  # compressed sparse column (CSC) format
end

"""
    _destroy!(p::Union{Ptr{MKLcsc}, Ptr{MKLcsr}})

Free the memory allocated by MKL for the sparse_matrix struct
"""
function _destroy!(p::Ptr{MKLcsc{T}}) where {T}
    ret = ccall((:mkl_sparse_destroy, libmkl_rt), sparse_status_t, (Ptr{MKLcsc{T}},), p)
    ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(string(ret)))
    nothing
end

"""
    MKLcsr{T<:MKLFloats}

The opaque struct `sparse_matrix` from the MKL library in compressed sparse row (CSR) format
"""
mutable struct MKLcsr{T<:MKLFloats}  # compressed sparse row (CSR) format
end

function _destroy!(p::Ptr{MKLcsr{T}}) where {T}
    ret = ccall((:mkl_sparse_destroy, libmkl_rt), sparse_status_t, (Ptr{MKLcsr{T}},), p)
    ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(ret))
    nothing
end

for (csc, exc, csr, T) in (
    (:mkl_sparse_s_create_csc, :mkl_sparse_s_export_csc, :mkl_sparse_s_create_csr, Float32,),
    (:mkl_sparse_d_create_csc, :mkl_sparse_d_export_csc, :mkl_sparse_d_create_csr, Float64,),
    (:mkl_sparse_c_create_csc, :mkl_sparse_c_export_csc, :mkl_sparse_c_create_csr, ComplexF32,),
    (:mkl_sparse_z_create_csc, :mkl_sparse_z_export_csc, :mkl_sparse_z_create_csr, ComplexF64,),
    )
    @eval begin
        function cscptr(m::SparseMatrixCSC{$T,BlasInt}) # create a Ptr{MKLcsc}
            r = Ref(Ptr{MKLcsc{$T}}(0))
            ret = ccall(
                ($(string(csc)), libmkl_rt), sparse_status_t,
                (Ref{Ptr{MKLcsc{$T}}}, sparse_index_base_t, BlasInt, BlasInt, Ptr{BlasInt},
                 Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}),
                r, SPARSE_INDEX_BASE_ONE, m.m, m.n, m.colptr, pointer(m.colptr, 2),
                m.rowval, m.nzval
            )
            ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(string(ret)))
            r[]
        end

        # create a SparseMatrixCSC from a Ptr{MKLcsc}
        function SparseArrays.SparseMatrixCSC(cscpt::Ptr{MKLcsc{$T}})
            indexing = Ref(Cint(0))
            rows = Ref(BlasInt(0))
            cols = Ref(BlasInt(0))
            rows_start = Ref(Ptr{BlasInt}(0))
            rows_end = Ref(Ptr{BlasInt}(0))
            col_indx = Ref(Ptr{BlasInt}(0))
            values = Ref(Ptr{$T}(0))
            ret = ccall(
                ($(string(exc)), libmkl_rt),
                sparse_status_t,
                (Ptr{MKLcsc{$T}}, Ref{Cint}, Ref{BlasInt}, Ref{BlasInt},
                Ref{Ptr{BlasInt}}, Ref{Ptr{BlasInt}}, Ref{Ptr{BlasInt}}, Ref{Ptr{$T}}),
                cscpt, indexing, rows, cols, rows_start, rows_end, col_indx, values,
            )
            ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(string(ret)))
            rowptr = Vector{BlasInt}(undef, rows[] + 1)
            unsafe_copyto!(pointer(rowptr), rows_start[], rows[])
            rowend = Vector{BlasInt}(undef, rows[])
            unsafe_copyto!(pointer(rowend), rows_end[], rows[])
            nonzeros = last(rowend) - indexing[]
            first(rowptr) == indexing[] || throw(ArgumentError("indexing = $indexing ≠ $(first(rowptr)) = first(rowptr)"))
            view(rowptr, 2:rows[]) == view(rowend, 1:(rows[]-1)) || throw(ArgumentError("indices are not dense"))
            rowptr[rows[] + 1] = last(rowend)
            colvals = Vector{BlasInt}(undef, nonzeros)
            unsafe_copyto!(pointer(colvals), col_indx[], nonzeros)
            nzvals = Vector{$T}(undef, nonzeros)
            unsafe_copyto!(pointer(nzvals), values[], nonzeros)
            SparseMatrixCSC{$T,BlasInt}(cols[], rows[], rowptr, colvals, nzvals)     
        end

        # create a Ptr{MKLcsr{T}} to the *transpose* of m
        function csrptrtr(m::SparseMatrixCSC{$T,BlasInt})
            r = Ref(Ptr{MKLcsr{$T}}(0))
            ret = ccall(
                ($(string(csr)), libmkl_rt), sparse_status_t,
                (Ref{Ptr{MKLcsr{$T}}}, sparse_index_base_t, BlasInt, BlasInt,
                Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}),
                r, SPARSE_INDEX_BASE_ONE, m.n, m.m, m.colptr, pointer(m.colptr, 2),
                m.rowval, m.nzval
            )
            ret == SPARSE_STATUS_SUCCESS || throw(ArgumentError(string(ret)))
            r[]
        end

    end # eval        
end #loop on types

"""
    struct matrix_descr

Mirror the C struct `matrix_desrc` in `mkl_spblas.h`
"""
struct matrix_descr
    type::sparse_matrix_type_t
    mode::sparse_fill_mode_t
    diag::sparse_diag_type_t
end

# special case the values
matrix_descr(A::LowerTriangular) = matrix_descr(SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT)
matrix_descr(A::UpperTriangular) = matrix_descr(SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT)
matrix_descr(A::UnitLowerTriangular) = matrix_descr(SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_UNIT)
matrix_descr(A::UnitUpperTriangular) = matrix_descr(SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_UNIT)
matrix_descr(A::Diagonal) = matrix_descr(SPARSE_MATRIX_TYPE_DIAGONAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT)
matrix_descr(A::Symmetric) = matrix_descr(SPARSE_MATRIX_TYPE_SYMMETRIC, (A.uplo == 'L' ? SPARSE_FILL_MODE_LOWER : SPARSE_FILL_MODE_UPPER), SPARSE_DIAG_NON_UNIT)
matrix_descr(A::Hermitian) = matrix_descr(SPARSE_MATRIX_TYPE_HERMITIAN, (A.uplo == 'L' ? SPARSE_FILL_MODE_LOWER : SPARSE_FILL_MODE_UPPER), SPARSE_DIAG_NON_UNIT)
matrix_descr(A::SparseMatrixCSC) = matrix_descr(SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT)
