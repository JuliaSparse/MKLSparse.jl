# Check dimension mismatch for C = A * B or C = A' * B
function matmul_error_check(A, B, C, t)
    if t == 'N'
        size(A, 2) == size(B, 1) || throw(DimensionMismatch())
        size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    else
        size(A, 2) == size(C, 1) || throw(DimensionMismatch())
        size(A, 1) == size(B, 1) || throw(DimensionMismatch())
    end
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
end

# Returns a suitable array C for in place A * B and A' * B
function get_suitable_array{T}(A, B::AbstractVector{T}, t)
    if t == 'N'
        return zeros(T, size(A,1))
    else
        return  zeros(T, size(A,2))
    end
end

function get_suitable_array{T}(A, B::AbstractMatrix{T}, t)
    if t == 'N'
        return zeros(T, size(A,1), size(B,2))
    else
        return zeros(T, size(B,2), size(A,1))
    end
end

# Generate mutating and unmutating A_mul_B and Ac_mul_B
for (rhs_type, mkl_func) in ((:StridedVector, :cscmv!),
                             (:StridedMatrix, :cscmm!))
    for (t_char, j_func, j_func_mut) in (('N', :(*), :A_mul_B!),
                                         ('T', :Ac_mul_B, :Ac_mul_B!))
        for m_type in (:(SparseMatrixCSC{T,BlasInt}),
                       :(Symmetric{T,SparseMatrixCSC{T,BlasInt}}),
                       :(LowerTriangular{T, SparseMatrixCSC{T,BlasInt}}),
                       :(UnitLowerTriangular{T, SparseMatrixCSC{T,BlasInt}}),
                       :(UpperTriangular{T, SparseMatrixCSC{T,BlasInt}}),
                       :(UnitUpperTriangular{T, SparseMatrixCSC{T,BlasInt}}))
            @eval begin
                # Generate mutating function
                function $(j_func_mut){T<:BlasFloat}(α::T, A::$(m_type),
                                                     B::$(rhs_type){T}, β::T,
                                                     C::$(rhs_type){T})
                    matmul_error_check(A, B, C, $(t_char))
                    $(mkl_func)($(t_char), α, matdescra(A),A,B,β,C)
                end

                # Generate non mutating function
                function $(j_func){T<:BlasFloat}(A::$(m_type),
                                                 B::$(rhs_type){T})
                    C = get_suitable_array(A, B, $(t_char))
                    $(j_func_mut)(one(T),A,B,zero(T), C)
                end
            end
        end
    end
end

# Generate mutating and unmutating A_ldiv_B and Ac_ldiv_Bc
for (rhs_type, mkl_func) in ((:StridedVector, :cscsv!),
                             (:StridedMatrix, :cscsm!))
    for (t_char, j_func, j_func_mut) in (('N', :(\), :A_ldiv_B!),
                                         ('T', :Ac_ldiv_B, :Ac_ldiv_B!))
        for m_type in (:(LowerTriangular{T, SparseMatrixCSC{T,BlasInt}}),
                       :(UnitLowerTriangular{T, SparseMatrixCSC{T,BlasInt}}),
                       :(UpperTriangular{T, SparseMatrixCSC{T,BlasInt}}),
                       :(UnitUpperTriangular{T, SparseMatrixCSC{T,BlasInt}}))
            @eval begin
                # Generate mutating function
                function $(j_func_mut){T<:BlasFloat}(α::T, A::$(m_type),
                                                     B::$(rhs_type){T},
                                                     C::$(rhs_type){T})
                    matmul_error_check(A, B, C, $(t_char))
                    $(mkl_func)($(t_char), α, matdescra(A), A.data, B, C)
                end

                # Generate non mutating function
                function $(j_func){T<:BlasFloat}(A::SparseMatrixCSC{T,BlasInt},
                                                 B::$(rhs_type){T})
                    C = get_suitable_array(A, B, $(t_char))
                    $(j_func_mut)(one(T),A,B, C)
                end
            end
        end
    end
end

