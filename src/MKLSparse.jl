module MKLSparse

    import Base.LinAlg: A_mul_B!, Ac_mul_B!, Ac_mul_B, *,
                        A_ldiv_B!, Ac_ldiv_B!, Ac_ldiv_B, \,
                        At_ldiv_B, At_ldiv_B!


    import Base.LinAlg: chksquare, factorize, show,
                        cholfact, ldltfact, factorize

    import Base.LinAlg: BlasFloat, BlasInt, DimensionMismatch,
                        UnitLowerTriangular, UnitUpperTriangular

    export matdescra

    _init_() = Base.blas_vendor() == :mkl || error("MKLSparse requires blas_vendor() == :mkl")

    include("./matdescra.jl")
    include("./generator.jl")

    include("./matmul.jl")
    include("./DSS/DSS.jl")

end # module
