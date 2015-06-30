# MKL_DSS_DEFAULTS
const MKL_DSS_DEFAULTS =  0


# Out-of-core level option definitions
const MKL_DSS_OOC_VARIABLE = 1024
const MKL_DSS_OOC_STRONG =   2048


# Refinement steps on / off
const MKL_DSS_REFINEMENT_OFF = 4096
const MKL_DSS_REFINEMENT_ON =  8192


# Solver step's substitution
const MKL_DSS_FORWARD_SOLVE   = 16384
const MKL_DSS_DIAGONAL_SOLVE  = 32768
const MKL_DSS_BACKWARD_SOLVE  = 49152
const MKL_DSS_TRANSPOSE_SOLVE = 262144
const MKL_DSS_CONJUGATE_SOLVE = 524288


# Single precision
const MKL_DSS_SINGLE_PRECISION = 65536


# Zero-based indexing
const MKL_DSS_ZERO_BASED_INDEXING = 131072


# Message level option definitions
const MKL_DSS_MSG_LVL_SUCCESS = -2147483647
const MKL_DSS_MSG_LVL_DEBUG   = -2147483646
const MKL_DSS_MSG_LVL_INFO    = -2147483645
const MKL_DSS_MSG_LVL_WARNING = -2147483644
const MKL_DSS_MSG_LVL_ERROR   = -2147483643
const MKL_DSS_MSG_LVL_FATAL   = -2147483642


# Termination level option definitions
const MKL_DSS_TERM_LVL_SUCCESS   = 1073741832
const MKL_DSS_TERM_LVL_DEBUG     = 1073741840
const MKL_DSS_TERM_LVL_INFO      = 1073741848
const MKL_DSS_TERM_LVL_WARNING   = 1073741856
const MKL_DSS_TERM_LVL_ERROR     = 1073741864
const MKL_DSS_TERM_LVL_FATAL     = 1073741872


# Structure option definitions
const MKL_DSS_SYMMETRIC                    = 536870976
const MKL_DSS_SYMMETRIC_STRUCTURE          = 536871040
const MKL_DSS_NON_SYMMETRIC                = 536871104
const MKL_DSS_SYMMETRIC_COMPLEX            = 536871168
const MKL_DSS_SYMMETRIC_STRUCTURE_COMPLEX  = 536871232
const MKL_DSS_NON_SYMMETRIC_COMPLEX        = 536871296


# Reordering option definitions
const MKL_DSS_AUTO_ORDER             = 268435520
const MKL_DSS_MY_ORDER               = 268435584
const MKL_DSS_OPTION1_ORDER          = 268435648
const MKL_DSS_GET_ORDER              = 268435712
const MKL_DSS_METIS_ORDER            = 268435776
const MKL_DSS_METIS_OPENMP_ORDER     = 268435840


# Factorization option definitions
const MKL_DSS_POSITIVE_DEFINITE           = 134217792
const MKL_DSS_INDEFINITE                  = 134217856
const MKL_DSS_HERMITIAN_POSITIVE_DEFINITE = 134217920
const MKL_DSS_HERMITIAN_INDEFINITE        = 134217984


const MKL_DSS_SUCCESS = 0
# Return status values
const RETURN_STATS = Dict{Int, ASCIIString}(
    -1  => "MKL_DSS_ZERO_PIVOT",
    -2  => "MKL_DSS_OUT_OF_MEMORY",
    -3  => "MKL_DSS_FAILURE",
    -4  => "MKL_DSS_ROW_ERR",
    -5  => "MKL_DSS_COL_ERR",
    -6  => "MKL_DSS_TOO_FEW_VALUES",
    -7  => "MKL_DSS_TOO_MANY_VALUES",
    -8  => "MKL_DSS_NOT_SQUARE",
    -9  => "MKL_DSS_STATE_ERR",
    -10 => "MKL_DSS_INVALID_OPTION",
    -11 => "MKL_DSS_OPTION_CONFLICT",
    -12 => "MKL_DSS_MSG_LVL_ERR",
    -13 => "MKL_DSS_TERM_LVL_ERR",
    -14 => "MKL_DSS_STRUCTURE_ERR",
    -15 => "MKL_DSS_REORDER_ERR",
    -16 => "MKL_DSS_VALUES_ERR",
    -17 => "MKL_DSS_STATISTICS_INVALID_MATRIX",
    -18 => "MKL_DSS_STATISTICS_INVALID_STATE",
    -19 => "MKL_DSS_STATISTICS_INVALID_STRING",
    -20 => "MKL_DSS_REORDER1_ERR",
    -21 => "MKL_DSS_PREORDER_ERR",
    -22 => "MKL_DSS_DIAG_ERR",
    -23 => "MKL_DSS_I32BIT_ERR",
    -24 => "MKL_DSS_OOC_MEM_ERR",
    -25 => "MKL_DSS_OOC_OC_ERR",
    -26 => "MKL_DSS_OOC_RW_ERR")


type DSSError <: Exception
    msg::AbstractString
end


macro errcheck(A)
    :(err = $A;  err == MKL_DSS_SUCCESS || throw(DSSError(RETURN_STATS[err])))
end
