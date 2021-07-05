using BenchmarkTools
using LightGraphs
using NetworkDynamics
using Random

include("benchmark_utils.jl")

const SUITE = BenchmarkGroup()

####
#### diffusion benchmarks
####
SUITE["diffusion"] = BenchmarkGroup(["homogeneous"])
vertex = diffusion_vertex()
edges = Dict("static_edge" => diffusion_edge(),
             "ode_edge" => diffusion_dedge())

for k in ["static_edge", "ode_edge"]
    SUITE["diffusion"][k] = BenchmarkGroup([k])
    SUITE["diffusion"][k]["assemble"] = BenchmarkGroup(["assemble"])
    SUITE["diffusion"][k]["call"] = BenchmarkGroup(["call"])
    SUITE["diffusion"][k]["call_mt"] = BenchmarkGroup(["call", "multithread"])
    edge = edges[k]

    for N ∈ [10, 100, 1_000, 10_000]  #, 100_000, 1_000_000]
        SUITE["diffusion"][k]["assemble"][N] = @benchmarkable begin
            network_dynamics($vertex, $edge, g)
        end setup = begin
            g = star_graph($N)
        end

        SUITE["diffusion"][k]["call"][N] = @benchmarkable begin
            nd(dx, x0, nothing, 0.0)
        end setup = begin
            g = star_graph($N)
            nd = network_dynamics($vertex, $edge, g)
            x0 = randn(length(nd.syms))
            dx = similar(x0)
        end

        SUITE["diffusion"][k]["call_mt"][N] = @benchmarkable begin
            nd(dx, x0, nothing, 0.0)
        end setup = begin
            g = star_graph($N)
            nd = network_dynamics($vertex, $edge, g; parallel=true)
            x0 = randn(length(nd.syms))
            dx = similar(x0)
        end
    end
end

####
#### kuramoto benchmarks
####
SUITE["kuramoto"] = BenchmarkGroup()

SUITE["kuramoto"]["homogeneous"] = BenchmarkGroup(["homogeneous"])
SUITE["kuramoto"]["heterogeneous"] = BenchmarkGroup(["heterogeneous"])

for k in ["homogeneous", "heterogeneous"]
    SUITE["kuramoto"][k]["assemble"] = BenchmarkGroup(["assemble"])
    SUITE["kuramoto"][k]["call"] = BenchmarkGroup(["call"])
    SUITE["kuramoto"][k]["call_mt"] = BenchmarkGroup(["call", "multithread"])
end

function homogeneous(N)
    rng = MersenneTwister(1)
    g = watts_strogatz(N, 3, 0.8, seed=1)
    edge = static_kuramoto_edge()
    vertex = kuramoto_vertex_2d()
    p = (randn(rng, nv(g)), randn(rng, ne(g)))
    (p, vertex, edge, g)
end

function heterogeneous(N)
    rng = MersenneTwister(1)
    g = watts_strogatz(N, 3, 0.8, seed=1)
    edge = static_kuramoto_edge()
    vertex = [kuramoto_vertex_1d(), kuramoto_vertex_2d()]
    vertices = vertex[shuffle(rng, vcat([1 for _ in 1:N÷2], [2 for _ in 1:N÷2]))]
    p = (randn(rng, nv(g)), randn(rng, ne(g)))
    (p, vertices, edge, g)
end

for N ∈ [10, 100, 1_000, 10_000]  #, 100_000, 1_000_000]
    # do both, for homogeneous and inhomogeneous system
    for f in [homogeneous, heterogeneous]
        name = string(f)
        SUITE["kuramoto"][name]["assemble"][N] = @benchmarkable begin
            network_dynamics(v, e, g)
        end setup = begin
            (p, v, e, g) = $f($N)
        end

        SUITE["kuramoto"][name]["call"][N] = @benchmarkable begin
            nd(dx, x0, p, 0.0)
        end setup = begin
            (p, v, e, g) = $f($N)
            nd = network_dynamics(v, e, g)
            x0 = randn(length(nd.syms))
            dx = similar(x0)
        end

        SUITE["kuramoto"][name]["call_mt"][N] = @benchmarkable begin
            nd(dx, x0, p, 0.0)
        end setup = begin
            (p, v, e, g) = $f($N)
            nd = network_dynamics(v, e, g; parallel=true)
            x0 = randn(length(nd.syms))
            dx = similar(x0)
        end
    end
end
