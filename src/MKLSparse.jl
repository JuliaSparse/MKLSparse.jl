module MKLSparse

    import Base.LinAlg: BlasFloat, BlasInt, DimensionMismatch,
                        UnitLowerTriangular, UnitUpperTriangular

    export matdescra

    _init_() = Base.blas_vendor() == :mkl || error("MKLSparse requires blas_vendor == :mkl")

    include("./matdescra.jl")
    include("./generator.jl")
    include("./matmul.jl")

end # module
