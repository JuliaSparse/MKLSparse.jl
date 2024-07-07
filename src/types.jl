# MKL sparse types

function mkl_type_specifier(T::Symbol)
    if T == :Float32
        's'
    elseif T == :Float64
        'd'
    elseif T == :ComplexF32
        'c'
    elseif T == :ComplexF64
        'z'
    else
        throw(ArgumentError("Unsupported numeric type $T"))
    end
end

function mkl_integer_specifier(INT::Symbol)
    if INT == :Int32
        ""
    elseif INT == :Int64
        "_64"
    else
        throw(ArgumentError("Unsupported numeric type $INT"))
    end
end

matrixdescra(A::LowerTriangular)     = matrix_descr('T','L','N')
matrixdescra(A::UpperTriangular)     = matrix_descr('T','U','N')
matrixdescra(A::Diagonal)            = matrix_descr('D','F','N')
matrixdescra(A::UnitLowerTriangular) = matrix_descr('T','L','U')
matrixdescra(A::UnitUpperTriangular) = matrix_descr('T','U','U')
matrixdescra(A::Symmetric)           = matrix_descr('S', A.uplo, 'N')
matrixdescra(A::Hermitian)           = matrix_descr('H', A.uplo, 'N')
matrixdescra(A::SparseMatrixCSC)     = matrix_descr('G', 'F', 'N')
matrixdescra(A::Transpose)           = matrixdescra(A.parent)
matrixdescra(A::Adjoint)             = matrixdescra(A.parent)

function Base.convert(::Type{sparse_operation_t}, trans::Char)
    if trans == 'N'
        SPARSE_OPERATION_NON_TRANSPOSE
    elseif trans == 'T'
        SPARSE_OPERATION_TRANSPOSE
    elseif trans == 'C'
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE
    else
        throw(ArgumentError("Unknown operation $trans"))
    end
end

function Base.convert(::Type{sparse_matrix_type_t}, mattype::Char)
    if mattype == 'G'
        SPARSE_MATRIX_TYPE_GENERAL
    elseif mattype == 'S'
        SPARSE_MATRIX_TYPE_SYMMETRIC
    elseif mattype == 'H'
        SPARSE_MATRIX_TYPE_HERMITIAN
    elseif mattype == 'T'
        SPARSE_MATRIX_TYPE_TRIANGULAR
    elseif mattype == 'D'
        SPARSE_MATRIX_TYPE_DIAGONAL
    else
        throw(ArgumentError("Unknown matrix type $mattype"))
    end
end

function Base.convert(::Type{sparse_matrix_type_t}, mattype::String)
    if mattype == "G"
        SPARSE_MATRIX_TYPE_GENERAL
    elseif mattype == "S"
        SPARSE_MATRIX_TYPE_SYMMETRIC
    elseif mattype == "H"
        SPARSE_MATRIX_TYPE_HERMITIAN
    elseif mattype == "T"
        SPARSE_MATRIX_TYPE_TRIANGULAR
    elseif mattype == "D"
        SPARSE_MATRIX_TYPE_DIAGONAL
    elseif mattype == "BT"
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR
    elseif mattype == "BD"
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL
    else
        throw(ArgumentError("Unknown matrix type $mattype"))
    end
end

function Base.convert(::Type{sparse_index_base_t}, index::Char)
    if index == 'Z'
        return SPARSE_INDEX_BASE_ZERO
    elseif index == 'O'
        return SPARSE_INDEX_BASE_ONE
    else
        throw(ArgumentError("Unknown index base $index"))
    end
end

function Base.convert(::Type{sparse_fill_mode_t}, uplo::Char)
    if uplo == 'L'
        SPARSE_FILL_MODE_LOWER
    elseif uplo == 'U'
        SPARSE_FILL_MODE_UPPER
    elseif uplo =='F'
        SPARSE_FILL_MODE_FULL
    else
        throw(ArgumentError("Unknown fill mode $uplo"))
    end
end

function Base.convert(::Type{sparse_diag_type_t}, diag::Char)
    if diag == 'U'
        SPARSE_DIAG_UNIT
    elseif diag == 'N'
       SPARSE_DIAG_NON_UNIT
    else
        throw(ArgumentError("Unknown diag type $diag"))
    end
end

function Base.convert(::Type{sparse_layout_t}, layout::Char)
    if layout == 'R'
        SPARSE_LAYOUT_ROW_MAJOR
    elseif layout == 'C'
        SPARSE_LAYOUT_COLUMN_MAJOR
    else
        throw(ArgumentError("Unknown layout $layout"))
    end
end

function Base.convert(::Type{verbose_mode_t}, verbose::String)
    if verbose == "off"
        SPARSE_VERBOSE_OFF
    elseif verbose == "basic"
        SPARSE_VERBOSE_BASIC
    elseif verbose == "extended"
        SPARSE_VERBOSE_EXTENDED
    else
        throw(ArgumentError("Unknown verbose mode $verbose"))
    end
end

function Base.convert(::Type{sparse_memory_usage_t}, memory::String)
    if memory == "none"
        SPARSE_MEMORY_NONE
    elseif memory == "aggressive"
        SPARSE_MEMORY_AGGRESSIVE
    else
        throw(ArgumentError("Unknown memory usage $memory"))
    end
end
