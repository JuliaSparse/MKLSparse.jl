# Basic types and constants for inspector-executor SpBLAS API

@enum(
    sparse_status_t,
    SPARSE_STATUS_SUCCESS           = 0,
    SPARSE_STATUS_NOT_INITIALIZED   = 1,
    SPARSE_STATUS_ALLOC_FAILED      = 2,
    SPARSE_STATUS_INVALID_VALUE     = 3,
    SPARSE_STATUS_EXECUTION_FAILED  = 4,
    SPARSE_STATUS_INTERNAL_ERROR    = 5,
    SPARSE_STATUS_NOT_SUPPORTED     = 6
)

@enum(
    sparse_operation_t,
    SPARSE_OPERATION_NON_TRANSPOSE       = 10,
    SPARSE_OPERATION_TRANSPOSE           = 11,
    SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12
)

@enum(
    sparse_matrix_type_t,
    SPARSE_MATRIX_TYPE_GENERAL            = 20,
    SPARSE_MATRIX_TYPE_SYMMETRIC          = 21,
    SPARSE_MATRIX_TYPE_HERMITIAN          = 22,
    SPARSE_MATRIX_TYPE_TRIANGULAR         = 23,
    SPARSE_MATRIX_TYPE_DIAGONAL           = 24,
    SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR   = 25,
    SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL     = 26
)

@enum(
    sparse_index_base_t,
    SPARSE_INDEX_BASE_ZERO    = 0,
    SPARSE_INDEX_BASE_ONE     = 1
)

@enum(
    sparse_fill_mode_t,
    SPARSE_FILL_MODE_LOWER  = 40,
    SPARSE_FILL_MODE_UPPER  = 41,
    SPARSE_FILL_MODE_FULL   = 42
)

@enum(
    sparse_diag_type_t,
    SPARSE_DIAG_NON_UNIT    = 50, 
    SPARSE_DIAG_UNIT        = 51
)

@enum(
    sparse_layout_t,
    SPARSE_LAYOUT_ROW_MAJOR    = 101,
    SPARSE_LAYOUT_COLUMN_MAJOR = 102
)

@enum(
    verbose_mode_t,
    SPARSE_VERBOSE_OFF      = 70,
    SPARSE_VERBOSE_BASIC    = 71,
    SPARSE_VERBOSE_EXTENDED = 72
)

@enum(
    sparse_memory_usage_t,
    SPARSE_MEMORY_NONE          = 80,
    SPARSE_MEMORY_AGGRESSIVE    = 81  
)

@enum(
    sparse_request_t,
    SPARSE_STAGE_FULL_MULT            = 90,
    SPARSE_STAGE_NNZ_COUNT            = 91,
    SPARSE_STAGE_FINALIZE_MULT        = 92,
    SPARSE_STAGE_FULL_MULT_NO_VAL     = 93,
    SPARSE_STAGE_FINALIZE_MULT_NO_VAL = 94
)

struct matrix_descr
    type::sparse_matrix_type_t
    mode::sparse_fill_mode_t
    diag::sparse_diag_type_t
end

matrix_descr(A::LowerTriangular) = matrix_descr(SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT)
matrix_descr(A::UpperTriangular) = matrix_descr(SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT)
matrix_descr(A::UnitLowerTriangular) = matrix_descr(SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_UNIT)
matrix_descr(A::UnitUpperTriangular) = matrix_descr(SPARSE_MATRIX_TYPE_TRIANGULAR, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_UNIT)
matrix_descr(A::Diagonal) = matrix_descr(SPARSE_MATRIX_TYPE_DIAGONAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT)
matrix_descr(A::Symmetric) = matrix_descr(SPARSE_MATRIX_TYPE_SYMMETRIC, (A.uplo == 'L' ? SPARSE_FILL_MODE_LOWER : SPARSE_FILL_MODE_UPPER), SPARSE_DIAG_NON_UNIT)
matrix_descr(A::Hermitian) = matrix_descr(SPARSE_MATRIX_TYPE_HERMITIAN, (A.uplo == 'L' ? SPARSE_FILL_MODE_LOWER : SPARSE_FILL_MODE_UPPER), SPARSE_DIAG_NON_UNIT)
matrix_descr(A::SparseMatrixCSC) = matrix_descr(SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT)
