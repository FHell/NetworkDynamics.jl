using CUDA
using LightGraphs
using NetworkDynamics
using BenchmarkTools

@inline function diffusion_edge!(e,v_s,v_d,p,t)
    @inbounds for i in 1:length(e)
        e[i] = v_s[i] - v_d[i]
    end
    nothing
end

@inline function diffusion_vertex_nd!(dv, v, edges, p, t)
    fill!(dv, 0)
    sum_coupling!(dv, edges)
    nothing
end

@inline function diffusion_vertex!(dv, v, agg::AbstractVector{<:AbstractFloat}, p, t)
    @inbounds for i in 1:length(dv)
        dv[i] = agg[i]
    end
    nothing
end

@inline function unsafe_add!(a, b)
    @inbounds for i in 1:length(a)
        a[i] += b[i]
    end
    nothing
end

struct Network{FloatMat, IdxMat, Vertex, Edge, Aggr}
    "vertex function (ODE)"
    vfun::Vertex
    "dimensions of edge function"
    vdim::Int
    "edge function (static)"
    efun::Edge
    "dimensions of vertex function"
    edim::Int
    "Aggregation function"
    aggfun::Aggr
    "Matrix(2, #edges), [srcidx, dstidx] per column"
    vidx_per_e::IdxMat
    "Matrix(?, #vertices), [e1idx, e2idx,...] of *incoming* edges per column"
    eidx_per_v::IdxMat
    "cache matrix for the aggregation"
    _aggregation::FloatMat
    "cache matrix for the static edges"
    _edge_u::FloatMat
end

function Network(; g::SimpleDiGraph, vf, vdim, ef, edim, aggf, AT=Array, FT=Float64)
    edge_u = Matrix{FT}(undef, edim, ne(g))
    aggregation = Matrix{FT}(undef, edim, nv(g))

    vidx_per_e = Matrix{Int32}(undef, 2, ne(g))
    for (i, e) in enumerate(edges(g))
        vidx_per_e[:, i] .= [e.src, e.dst]
    end

    # indices of the outgoing edges
    # i.e. find idx for occurences of v in the dst row of vidx_per_edge
    eidx_per_v = zeros(Int, maximum(indegree(g)), nv(g))
    for v in 1:nv(g)
        eidx = findall(isequal(v), vidx_per_e[2, :])
        eidx_per_v[1:length(eidx), v] .= eidx
    end

    Network(vf, vdim, ef, edim, aggf,
            AT{Int32,2}(vidx_per_e),
            AT{Int32,2}(eidx_per_v),
            AT{FT,2}(aggregation),
            AT{FT,2}(edge_u))
end

LightGraphs.ne(nw::Network) = size(nw.vidx_per_e, 2)
LightGraphs.nv(nw::Network) = size(nw.eidx_per_v, 2)

function (nw::Network{<:Matrix})(du, u, p, t)
    # calculate edges
    @inbounds for eidx in 1:ne(nw)
        eu = view(nw._edge_u, :, eidx)
        vs = view(u, :, nw.vidx_per_e[1, eidx])
        vd = view(u, :, nw.vidx_per_e[2, eidx])
        nw.efun(eu, vs, vd, p, t)
    end

    # aggregate edge values
    fill!(nw._aggregation, 0)
    @inbounds for vidx in 1:nv(nw)
        agg = view(nw._aggregation, :, vidx)
        for eidx in view(nw.eidx_per_v, :, vidx)
            eidx == 0 && break # no more edges left
            values = view(nw._edge_u, :, eidx)
            nw.aggfun(agg, values)
        end
    end

    # calculate vertex functions (needs  fill!(du, 0) ? )
    @inbounds for vidx in 1:nv(nw)
        vdu = view(du, :, vidx)
        vu  = view(u, :, vidx)
        agg = view(nw._aggregation, :, vidx)
        nw.vfun(vdu, vu, agg, p, t)
    end
end

function edge_kernel!(efun, _edge_u, u, p, t, N, vidx_per_e)
    eidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if eidx <= N
        eu = view(_edge_u, :, eidx)
        vs = view(u, :, vidx_per_e[1, eidx])
        vd = view(u, :, vidx_per_e[2, eidx])
        efun(eu, vs, vd, nothing, t)
    end
    return nothing
end

function aggregation_kernel!(aggfun, _aggregation, _edge_u, N, eidx_per_v)
    vidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if vidx <= N
        agg = view(_aggregation, :, vidx)
        for eidx in view(eidx_per_v, :, vidx)
            eidx == 0 && break # no more edges left
            values = view(_edge_u, :, eidx)
            aggfun(agg, values)
        end
    end
    return nothing
end

function vertex_kernel!(vfun, du, u, _aggregation, p, t, N)
    vidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if vidx <= N
        vdu = view(du, :, vidx)
        vu  = view(u, :, vidx)
        agg = view(_aggregation, :, vidx)
        vfun(vdu, vu, agg, p, t)
    end
end

function (nw::Network{<:CuMatrix})(du, u, p, t)
    # edge loop
    Ne, Nv = ne(nw), nv(nw)
    numb = ceil(Int, Ne/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numb edge_kernel!(nw.efun, nw._edge_u,
                                                   u, p, t,
                                                   Ne, nw.vidx_per_e)
    end

    # aggregate edge values
    numb = ceil(Int, Nv/256)
    fill!(nw._aggregation, 0)
    CUDA.@sync begin
        @cuda threads=256 blocks=numb aggregation_kernel!(nw.aggfun,
                                                          nw._aggregation,
                                                          nw._edge_u,
                                                          Nv, nw.eidx_per_v)
    end

    # calculate vertex functions (needs  fill!(du, 0) ? )
    CUDA.@sync begin
        @cuda threads=256 blocks=numb vertex_kernel!(nw.vfun,
                                                     du, u,
                                                     nw._aggregation,
                                                     p, t, Nv)
    end
    return nothing
end

####
#### lets benchmark nd, the GPU and the CPU verions
####
Nv = Int[]
tnd = Float64[]
tcpu = Float64[]
tgpu = Float64[]
for N in 1_000:1_000:10_000
    @info "N = $N"
    g = SimpleDiGraph(watts_strogatz(N, 4, 0.5))
    vdim = 1
    edim = 1
    odevertex = ODEVertex(f! = diffusion_vertex_nd!,dim = vdim)
    staticedge = StaticEdge(f! = diffusion_edge!, dim = edim, coupling=:directed)

    nd = network_dynamics(odevertex, staticedge, g)
    nw_cpu = Network(; g, vf=diffusion_vertex!, vdim, ef=diffusion_edge!, edim, aggf=unsafe_add!)
    nw_gpu = Network(; g, vf=diffusion_vertex!, vdim, ef=diffusion_edge!, edim, aggf=unsafe_add!, AT=CuArray)

    u_cpu = rand(vdim, nv(g));
    du_cpu = zeros(size(u_cpu)...);
    u_gpu = CuArray(u_cpu);
    du_gpu = CuArray(du_cpu);
    u_nd  = u_cpu[:]; # copy and bring in right shape
    du_nd = du_cpu[:];

    nd(du_nd, u_nd, nothing, 0.0)
    nw_cpu(du_cpu, u_cpu, nothing, 0.0)
    nw_gpu(du_gpu, u_gpu, nothing, 0.0)

    # du_nd .- du[:]
    @assert du_nd ≈ du_cpu[:]
    @assert du_nd ≈ Array(du_gpu[:])

    nd = @benchmark $nd($du_nd, $u_nd, nothing, 0.0)
    cpu = @benchmark $nw_cpu($du_cpu, $u_cpu, nothing, 0.0)
    gpu = @benchmark $nw_gpu($du_gpu, $u_gpu, nothing, 0.0)

    push!(Nv, N)
    push!(tnd, median(nd).time)
    push!(tcpu, median(cpu).time)
    push!(tgpu, median(gpu).time)
end

using UnicodePlots
using Plots
#results
Nv = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
tnd =  [31036.0, 64741.0, 100002.0, 136430.5, 169399.0, 208196.0, 245205.0, 284399.0, 317361.0, 352364.0]
tcpu = [18987.0, 39246.0, 59190.0, 78887.0, 98959.0, 119536.0, 141710.0, 161325.0, 183985.0, 204158.0]
tgpu = [27177.5, 24925.0, 26865.0, 24559.0, 24740.5, 25134.0, 25101.0, 25380.5, 25658.0, 26085.0]

p = lineplot(log.(Nv), log.(tnd); name="nd", xlabel="log(N)", ylabel="log(t)", title="coreloop timings")
lineplot!(p, log.(Nv), log.(tcpu); name="CPU")
lineplot!(p, log.(Nv), log.(tgpu); name="GPU")

using OrdinaryDiffEq
using DiffEqGPU

Nv = Int[]
tnd = Float64[]
tcpu = Float64[]
tgpu = Float64[]
for N in [10, 100, 1_000, 5_000, 10_000]
    @info "N = $N"

    g = SimpleDiGraph(watts_strogatz(N, 4, 0.5))
    vdim = 1
    edim = 1
    odevertex = ODEVertex(f! = diffusion_vertex_nd!,dim = vdim)
    staticedge = StaticEdge(f! = diffusion_edge!, dim = edim, coupling=:directed)

    nd = network_dynamics(odevertex, staticedge, g)
    nw_cpu = Network(; g, vf=diffusion_vertex!, vdim, ef=diffusion_edge!, edim, aggf=unsafe_add!)
    nw_gpu = Network(; g, vf=diffusion_vertex!, vdim, ef=diffusion_edge!, edim, aggf=unsafe_add!, AT=CuArray)

    u0_cpu = rand(vdim, nv(g));
    u0_gpu = CuArray(u0_cpu);
    u0_nd  = u0_cpu[:]; # copy and bring in right shape

    tspan = (0., 100.)
    prob_nd = ODEProblem(nd, u0_nd, tspan)
    prob_cpu = ODEProblem(nw_cpu, u0_cpu, tspan)
    prob_gpu = ODEProblem(nw_gpu, u0_gpu, tspan)

    tsit_nd = solve(prob_nd, Tsit5());
    tsit_cpu = solve(prob_cpu, Tsit5());
    tsit_gpu = solve(prob_gpu, Tsit5());

    @assert tsit_nd[end] ≈ tsit_cpu[end][:]
    @assert tsit_nd[end] ≈ Array(tsit_gpu[end])[:]

    nd =  @benchmark $solve($prob_nd, $Tsit5())
    cpu = @benchmark $solve($prob_cpu, $Tsit5())
    gpu = @benchmark $solve($prob_gpu, $Tsit5())

    push!(Nv, N)
    push!(tnd, median(nd).time)
    push!(tcpu, median(cpu).time)
    push!(tgpu, median(gpu).time)
end
# results
Nv = [10, 100, 1000, 5000, 10000]
tnd = [428508.0, 4.532003e6, 6.292465e7, 3.43278027e8, 8.17074963e8]
tcpu = [280178.0, 2.562238e6, 3.87406555e7, 1.98708704e8, 4.57421916e8]
tgpu = [5.7550438e7, 7.3006628e7, 9.169478e7, 9.8668921e7, 1.10475052e8]

p = lineplot(log.(Nv), log.(tnd); name="nd", xlabel="log(N)", ylabel="log(t)", title="Tsit5() timings")
lineplot!(p, log.(Nv), log.(tcpu); name="CPU")
lineplot!(p, log.(Nv), log.(tgpu); name="GPU")
