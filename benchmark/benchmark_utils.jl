####
#### Diffusion system
####
Base.@propagate_inbounds function diffusionedge!(e, v_s, v_d, _, _)
    e[1] = v_s[1] - v_d[1]
    nothing
end
diffusion_edge() = StaticEdge(f! = diffusionedge!, dim =1, coupling = :antisymmetric)

Base.@propagate_inbounds function diffusion_dedge!(de, e, v_s, v_d, _, _)
    de[1] = 100. * (sin(v_s[1] - v_d[1]) - e[1])
    nothing
end
diffusion_dedge() = ODEEdge(f! = diffusion_dedge!, dim = 1, coupling = :undirected)

Base.@propagate_inbounds function diffusionvertex!(dv, _, edges, _, _)
    dv[1] = 0.
    @inbounds for e in edges
        dv[1] += e[1]
    end
    nothing
end
diffusion_vertex() = ODEVertex(f! = diffusionvertex!, dim = 1)

####
#### inhomogenious kuramoto system
####
Base.@propagate_inbounds function kuramoto_edge!(e, θ_s, θ_d, K, t)
    e[1] = K * sin(θ_s[1] - θ_d[1])
end
static_kuramoto_edge() = StaticEdge(f! = kuramoto_edge!, dim = 1, coupling = :antisymmetric)

Base.@propagate_inbounds function kuramoto_vertex!(dθ, θ, edges, ω, t)
    dθ[1] = ω
    sum_coupling!(dθ, edges)
end
kuramoto_vertex_1d() = ODEVertex(f! = kuramoto_vertex!, dim = 1, sym=[:θ])

Base.@propagate_inbounds function kuramoto_inertia!(dv, v, edges, P, t)
    dv[1] = v[2]
    dv[2] = P - 1. * v[2]
    @inbounds for e in edges
        dv[2] += e[1]
    end
end
kuramoto_vertex_2d() = ODEVertex(f! = kuramoto_inertia!, dim = 2, sym= [:θ, :ω]);
