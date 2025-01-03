# faster copytri!() that uses @simd, @inbounds
# TODO copy by blocks to minimize cache misses
@inline function fastcopytri!(f, A::AbstractMatrix{<:BlasFloat}, uplo::AbstractChar)
    n = LinearAlgebra.checksquare(A)
    if uplo == 'U'
        @inbounds for i in axes(A, 1)
            @simd for j in (i+1):n
                A[j,i] = f(A[i,j])
            end
        end
    elseif uplo == 'L'
        @inbounds for i in axes(A, 1)
            @simd for j in (i+1):n
                A[i,j] = f(A[j,i])
            end
        end
    else
        throw(ArgumentError(lazy"uplo argument must be 'U' (upper) or 'L' (lower), got $uplo"))
    end
    return A
end

fastcopytri!(A::AbstractMatrix, uplo::AbstractChar, conjugate::Bool = false) =
    fastcopytri!(conjugate ? conj : identity, A, uplo)
