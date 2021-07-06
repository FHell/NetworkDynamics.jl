using CUDA
using BenchmarkTools
using UnicodePlots

####
#### most simple case, linear data
####

function staticedge!(e, vs, vd)
    @inbounds e[1] = sin(vs[1] - vd[1])
    return nothing
end

function loop(e::T, vs::T, vd::T) where {T<:Array}
    for i in 1:length(vs)
        staticedge!(view(e, i), view(vs, i), view(vd, i))
    end
    return nothing
end

function edge_kernel!(e, vs, vd, length)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= length
        staticedge!(view(e, index), view(vs, index), view(vd, index))
    end
    return nothing
end

function loop(e::T, vs::T, vd::T) where {T<:CuArray}
    N = length(e)
    numblocks = ceil(Int, N/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks edge_kernel!(e, vs, vd, N)
    end
    return nothing
end

Nv = Int[]
tcpu = Float64[]
tgpu = Float64[]

for N in [100, 500, 1000, 5000]
    println(N)
    vs = rand(Float32, N);
    vd = rand(Float32, N);
    e = zeros(Float32, N);
    vs_d = CuArray(vs);
    vd_d = CuArray(vd);
    e_d = CuArray(e);

    @assert Array(e_d) == e
    @assert Array(vs_d) == vs
    @assert Array(vd_d) == vd

    loop(e, vs, vd)
    loop(e_d, vs_d, vd_d)
    @assert Array(e_d) ≈ e

    cpu = @benchmark loop($e, $vs, $vd)
    gpu = @benchmark loop($e_d, $vs_d, $vd_d)
    # BenchmarkTools.judge(median(gpu), median(cpu))

    push!(Nv, N)
    push!(tcpu, median(cpu).time)
    push!(tgpu, median(gpu).time)
end

p = lineplot(Nv, log.(tcpu); name="CPU")
lineplot!(p, Nv, log.(tgpu); name="GPU")
