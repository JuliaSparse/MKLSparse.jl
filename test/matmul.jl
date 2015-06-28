ones100 = ones(100)
onesthou = ones(1000)
dA = full(sA)
dTl = full(sTl)
dTu = full(sTu)

@test_approx_eq(sA*ones100,dA*ones100)
@test_approx_eq(sTl*ones100,dTl*ones100)
@test_approx_eq(sTu*ones100,dTu*ones100)

@test_approx_eq(sA'onesthou,dA'onesthou)

@test_approx_eq(LowerTriangular(sTl)*ones100,sTl*ones100)
@test_approx_eq(UpperTriangular(sTu)*ones100,sTu*ones100)
@test_approx_eq(Base.LinAlg.UnitLowerTriangular(sTl)*ones100,
                Base.LinAlg.UnitLowerTriangular(dTl)*ones100)
@test_approx_eq(Base.LinAlg.UnitUpperTriangular(sTu)*ones100,
                Base.LinAlg.UnitUpperTriangular(dTu)*ones100)

@test_approx_eq(Symmetric(sTl,:L)*ones100,Symmetric(dTl,:L)*ones100)

@test_approx_eq(LowerTriangular(sTl)\ones100,dTl\ones100)
@test_approx_eq(A_ldiv_B!(1.0,LowerTriangular(sTl),ones100,similar(ones100)),dTl\ones100)
