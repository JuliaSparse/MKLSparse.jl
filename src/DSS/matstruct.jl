import Base: issym, ishermitian

immutable MatrixSymStructure
    symmetric::Bool
    hermitian::Bool
    chol_candidate::Bool
end

issym(mss::MatrixSymStructure) = mss.symmetric
ishermitian(mss::MatrixSymStructure) = mss.hermitian
ischolcand(mss::MatrixSymStructure) = mss.chol_candidate

function MatrixSymStructure(A::SparseMatrixCSC)
    hermitian = true
    chol_candidate = true
    symmetric = true

    n = Base.LinAlg.chksquare(A)

    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    tracker = copy(A.colptr)
    @inbounds for col = 1:n
        for p = tracker[col]:colptr[col+1]-1
            val = nzval[p]
            if val == 0; continue; end # In case of explicit zeros
            row = rowval[p]
            if row < col
                return MatrixSymStructure(false, false, false)
            elseif row == col # Diagonal element
                if imag(val) != 0
                    hermitian = false # Hermitians have real diagonal
                end
                if real(val) < 0 || imag(val) != 0 # Cholesky candidates have real pos diag
                    chol_candidate = false
                end
            else
                row2 = rowval[tracker[row]]
                val2 = nzval[tracker[row]]
                if row2 > col
                    return MatrixSymStructure(false, false, false)
                end
                if row2 == col # A[i,j] and A[j,i] exists
                    if val != val2
                        symmetric = false
                    end
                    if val != conj(val2)
                        hermitian = false
                    end
                    tracker[row] += 1
                end
            end
        end
        # check if we can abort early
        if symmetric == false && hermitian == false && chol_candidate == false
            return MatrixSymStructure(false, false, false)
        end
    end
    return MatrixSymStructure(symmetric, hermitian, chol_candidate)
end
