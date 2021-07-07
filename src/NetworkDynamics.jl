module NetworkDynamics

using Reexport
using DiffEqBase
using LightGraphs

include("Utilities.jl")
@reexport using .Utilities

include("ComponentFunctions.jl")
@reexport using .ComponentFunctions

include("NetworkStructures.jl")
@reexport using .NetworkStructures

include("nd_ODE_ODE.jl")
@reexport using .nd_ODE_ODE_mod

include("nd_ODE_Static.jl")
@reexport using .nd_ODE_Static_mod

include("nd_DDE_Static.jl")
@reexport using .nd_DDE_Static_mod


export network_dynamics



"""
This function assembles the arrays that hold the structural information
of the individual vertices and edges: Their dimensions, the symbols of their
variables and their mass matrices. Used internally by the network_dynamics
constructors.
"""
function collect_ve_info(vertices!, edges!, graph)
    if vertices! isa Array
        @assert length(vertices!) == nv(graph)
        v_dims = Int[v.dim for v in vertices!]
        symbols_v = [Symbol(vertices![i].sym[j],"_",i)
                           for i in 1:length(vertices!)
                           for j in 1:v_dims[i]]
        mmv_array = [v.mass_matrix for v in vertices!]
    else
        v_dims = [vertices!.dim for v in vertices(graph)]
        symbols_v = [Symbol(vertices!.sym[j],"_",i)
                     for i in 1:nv(graph)
                     for j in 1:v_dims[i]]
        mmv_array = [vertices!.mass_matrix for v in vertices(graph)]
    end

    if edges! isa Array
        @assert length(edges!) == ne(graph)
        e_dims = Int[e.dim for e in edges!]
        symbols_e = [Symbol(edges![i].sym[j],"_",i)
                           for i in 1:length(edges!)
                           for j in 1:e_dims[i]]
        if eltype(edges!)  <: Union{StaticEdge, StaticDelayEdge}  # improve type hierarchy
            mme_array = nothing
        else
            mme_array = [e.mass_matrix for e in edges!]
        end
    else
        e_dims = [edges!.dim for e in edges(graph)]
        symbols_e = [Symbol(edges!.sym[j],"_",i)
                     for i in 1:ne(graph)
                     for j in 1:e_dims[i]]
        if typeof(edges!) <: Union{StaticEdge, StaticDelayEdge} # improve type hierarchy
            mme_array = nothing
        else
            mme_array = [edges!.mass_matrix for e in edges(graph)]
        end
    end

    v_dims, e_dims, symbols_v, symbols_e, mmv_array, mme_array
end


function collect_unique_components(ver)
    uver = Tuple(unique(ver))
    ind_uver = Vector{Int64}[]
    for v in uver
        ind_v = Int64[]
        for i in eachindex(ver)
            if v == ver[i]
                push!(ind_v, i)
            end
        end
        push!(ind_uver, ind_v)
    end
    uver, ind_uver
end

"""
    network_dynamics(vertices!, edges!, g; parallel = false)

Assembles the the dynamical equations of the network problem into an `ODEFunction`
compatible with the `DifferentialEquations.jl` solvers. Takes as arguments an array
of VertexFunctions **`vertices!`**, an array of EdgeFunctions **`edges!`** and a
`LightGraph.jl` object **`g`**. The optional argument `parallel` is a boolean
value that denotes if the central loop should be executed in parallel with the number of threads set by the environment variable `JULIA_NUM_THREADS`.
"""
function network_dynamics(vertices!::Union{Array{T, 1}, T},
                          edges!::Union{Array{U, 1}, U},
                          graph;
                          x_prototype=zeros(1),
                          parallel=false) where {T <: ODEVertex, U <: StaticEdge}

    warn_parallel(parallel)

    # user_edges! = copy(edges!)
    edges! = prepare_edges(edges!, graph)


    v_dims, e_dims, symbols_v, symbols_e, mmv_array, mme_array = collect_ve_info(vertices!, edges!, graph)

    # These arrays are used for initializing the GraphData and will be overwritten
    v_array = similar(x_prototype, sum(v_dims))
    e_array = similar(x_prototype, sum(e_dims))

    symbols = symbols_v

    unique_vertices!, unique_v_indices = collect_unique_components(vertices!)

    graph_stucture = GraphStruct(graph, v_dims, e_dims, symbols_v, symbols_e)

    graph_data = GraphData(v_array, e_array, graph_stucture)

    nd! = nd_ODE_Static(vertices!, unique_vertices!, unique_v_indices, edges!, graph, graph_stucture, graph_data, parallel)
    mass_matrix = construct_mass_matrix(mmv_array, graph_stucture)

    ODEFunction(nd!; mass_matrix = mass_matrix, syms=symbols)
end

## DDE

function network_dynamics(vertices!::Union{Array{T, 1}, T}, edges!::Union{Array{U, 1}, U}, graph; initial_history=nothing, x_prototype=zeros(1), parallel=false) where {T <: DDEVertex, U <: StaticDelayEdge}
    warn_parallel(parallel)

    edges! = prepare_edges(edges!, graph)

    v_dims, e_dims, symbols_v, symbols_e, mmv_array, mme_array = collect_ve_info(vertices!, edges!, graph)

    # These arrays are used for initializing the GraphData and will be overwritten
    v_array = similar(x_prototype, sum(v_dims))
    e_array = similar(x_prototype, sum(e_dims))

    # default
    if initial_history === nothing
        initial_history = ones(sum(v_dims))
    end

    symbols = symbols_v

    graph_stucture = GraphStruct(graph, v_dims, e_dims, symbols_v, symbols_e)

    graph_data = GraphData(v_array, e_array, graph_stucture)

    nd! = nd_DDE_Static(vertices!, edges!, graph, graph_stucture, graph_data, initial_history, parallel)
    mass_matrix = construct_mass_matrix(mmv_array, graph_stucture)

    DDEFunction(nd!; mass_matrix = mass_matrix, syms=symbols)
end

"""
Promotes StaticEdge to StaticDelayEdge if there is a DDEVertex
"""
function network_dynamics(vertices!::Union{Array{T, 1}, T}, edges!::Union{Array{U, 1}, U}, graph; initial_history=nothing, x_prototype=zeros(1), parallel=false) where {T <: DDEVertex, U <: StaticEdge}
    if edges! isa Array
        network_dynamics(vertices!, Array{StaticDelayEdge}(edges!), graph, initial_history = initial_history, x_prototype =  x_prototype, parallel = parallel)
    else
        network_dynamics(vertices!, StaticDelayEdge(edges!), graph, initial_history = initial_history, x_prototype =  x_prototype, parallel = parallel)
    end
end
"""
Promotes ODEVertex to DDEVertex if there is a StaticDelayEdge
"""
function network_dynamics(vertices!::Union{Array{T, 1}, T}, edges!::Union{Array{U, 1}, U}, graph; initial_history=nothing, x_prototype=zeros(1), parallel=false) where {T <: ODEVertex, U <: StaticDelayEdge}
    if vertices! isa Array
        network_dynamics(Array{DDEVertex}(vertices!), edges!, graph, initial_history = initial_history, x_prototype =  x_prototype, parallel = parallel)
    else
        network_dynamics(DDEVertex(vertices!), edges!, graph, initial_history = initial_history, x_prototype =  x_prototype, parallel = parallel)
    end
end

## ODE

function network_dynamics(vertices!::Union{Array{T, 1}, T}, edges!::Union{Array{U, 1}, U}, graph; x_prototype=zeros(1), parallel=false) where {T <: ODEVertex, U <: ODEEdge}

    warn_parallel(parallel)

    # We are abusing prepare_edges here to throw errors for impossible connections of
    # coupling type and graph. We don't save it's return value though, since reconstruction
    # of ODEEdges is forbidden at the moment.

    prepare_edges(edges!, graph)

    v_dims, e_dims, symbols_v, symbols_e, mmv_array, mme_array = collect_ve_info(vertices!, edges!, graph)

    # These arrays are used for initializing the GraphData and will be overwritten
    x_array = similar(x_prototype, sum(v_dims) + sum(e_dims))
    v_array = view(x_array, 1:sum(v_dims))
    e_array = view(x_array, sum(v_dims)+1:sum(v_dims)+sum(e_dims))

    symbols = vcat(symbols_v, symbols_e)

    graph_stucture = GraphStruct(graph, v_dims, e_dims, symbols_v, symbols_e)

    graph_data = GraphData(v_array, e_array, graph_stucture)

    nd! = nd_ODE_ODE(vertices!, edges!, graph, graph_stucture, graph_data, parallel)

    mass_matrix = construct_mass_matrix(mmv_array, mme_array, graph_stucture)

    ODEFunction(nd!; mass_matrix = mass_matrix, syms=symbols)
end

function network_dynamics(vertices!,  edges!, graph; parallel=false)
    # If vertices! and/or edges! are individual functions and no other dispatch was
    # triggered, assume all vertices, respectively edges will be of that type
    if typeof(vertices!) <: VertexFunction
        vertices! = Array{VertexFunction}([vertices! for i in 1:nv(graph)])
    end
    if typeof(edges!) <: EdgeFunction
        edges! = Array{EdgeFunction}([edges! for i in 1:ne(graph)])
    end

    try
        Array{VertexFunction}(vertices!)
    catch err
        throw(ArgumentError("Cannot convert the vertices to an Array{VertexFunction}!"))
    end

    try
        Array{EdgeFunction}(edges!)
    catch err
        throw(ArgumentError("Cannot convert the edges to an Array{EdgeFunction}!"))
    end
    va! = Array{VertexFunction}(vertices!)
    ea! = Array{EdgeFunction}(edges!)
    network_dynamics(va!, ea!, graph, parallel = parallel)
end

function network_dynamics(vertices!::Array{VertexFunction}, edges!::Array{EdgeFunction}, graph; kwargs...)
    @assert length(vertices!) == nv(graph)
    @assert length(edges!) == ne(graph)

    contains_delay = false
    contains_dyn_edge = false

    for e in edges!
        if isa(e, StaticDelayEdge)
            contains_delay = true
        end
        if isa(e, ODEEdge)
            contains_dyn_edge = true
        end
    end
    for v in vertices!
        if isa(v, DDEVertex)
            contains_delay = true
        end
    end

    contains_delay && contains_dyn_edge ? error(
        ArgumentError("ODEEdges with delay are not supported at the moment.")) : nothing

    # If one Edge or Vertex needs access to the history function, all network components are promoted to hisotry aware version -> lots more variables that are potentially not used are passed around. the multilayer structure should (partially) work around that need.
    if contains_delay
        return network_dynamics(Array{DDEVertex}(vertices!),Array{StaticDelayEdge}(edges!), graph; kwargs...)
    end

    # If one edge is an ODEEdge all other edges will be promoted. This should be
    # solved more elegantly by the upcoming multilayer structure.

    if contains_dyn_edge
        return network_dynamics(Array{ODEVertex}(vertices!),Array{ODEEdge}(edges!), graph; kwargs...)
    else
        return network_dynamics(Array{ODEVertex}(vertices!),Array{StaticEdge}(edges!), graph; kwargs...)
    end
    nothing
end



# Here is not the best place for this error,
# but requires the least changes before the next refactor.
function prepare_edges(edges, g)
    @assert typeof(edges) <: Union{EdgeFunction, Vector}
    throw(ArgumentError("Graph type not recognized. Currently only SimpleGraph and SimpleDiGraph are supported."))
end
"""
If only a sinlge Function is given, not an Array of EdgeFunctions.
"""
function prepare_edges(edge::EdgeFunction, g::SimpleGraph)
    if edge.coupling == :directed
        throw(ArgumentError("Coupling type of EdgeFunction not available for undirected Graphs"))
    elseif edge.coupling == :undefined
        @info("Reconstructing EdgeFunction with :undefined coupling type...")
        return reconstruct_edge(edge, :undirected)
    end
    return edge
end

function prepare_edges(edge::EdgeFunction, g::SimpleDiGraph)
    if edge.coupling ∈ (:symmetric, :antisymmetric, :undirected, :fiducial)
        throw(ArgumentError("Coupling type of EdgeFunction not available for directed Graphs"))
    elseif edge.coupling == :undefined
        @info("Reconstructing EdgeFunction with :undefined coupling type...")
        return reconstruct_edge(edge, :directed)
    end
    return edge
end

""" prepare_edges(edges, g::SimpleGraph)


"""
function prepare_edges(edges::Vector, g::SimpleGraph)
    # Depending on the coupling we might get different eltypes
    new_edges = Vector{EdgeFunction}(undef, length(edges))
    infobool = true
    for (i, edge) in enumerate(edges)
        if edge.coupling == :directed
            throw(ArgumentError("Coupling type of edge $i not available for undirected Graphs"))
        elseif edge.coupling == :undefined
            if infobool
                @info("Reconstructing EdgeFuntions with :undefined coupling type.")
                infobool = false
            end
            new_edges[i] = reconstruct_edge(edge, :undirected)
        else
            new_edges[i] = edges[i]
        end
    end
    # by recreating the array the eltype gets narrowed down as much as possible
    narrowed_type = [e for e in new_edges]
    return narrowed_type
end

"""
"""
function prepare_edges(edges::Vector, g::SimpleDiGraph)
    # Depending on the coupling we might get different eltypes
    new_edges = Vector{EdgeFunction}(undef, length(edges))
    infobool = true
    for (i, edge) in enumerate(edges)
        if edge.coupling ∈ (:symmetric, :antisymmetric, :undirected, :fiducial)
            throw(ArgumentError("Coupling type of edge $i not available for directed Graphs"))
        elseif edge.coupling == :undefined
            if infobool
                @info("Reconstructing EdgeFuntions with :undefined coupling type.")
                infobool = false
            end
            new_edges[i] = reconstruct_edge(edge, :directed)
        else
            new_edges[i] = edges[i]
        end
    end
    # by recreating the array the eltype gets narrowed down as much as possible
    narrowed_type = [e for e in new_edges]
    return narrowed_type
end




@inline function reconstruct_edge(edge::StaticEdge, coupling::Symbol)
    let f! = edge.f!, dim = edge.dim, sym = edge.sym
        return StaticEdge(f! = f!,
                          dim = dim,
                          coupling = coupling,
                          sym = sym)
    end
end
@inline function reconstruct_edge(edge::StaticDelayEdge, coupling::Symbol)
    let f! = edge.f!, dim = edge.dim, sym = edge.sym
        return StaticDelayEdge(f! = f!,
                               dim = dim,
                               coupling = coupling,
                               sym = sym)
    end
end
@inline function reconstruct_edge(edge::ODEEdge, coupling::Symbol)
    error("Reconstruction of ODEEdges is not implemented at the moment.")
end

# Not used at the moment
#
# @inline function reconstruct_edge(edge::ODEEdge, coupling::Symbol)
#     let f! = edge.f!, dim = edge.dim, sym = edge.sym, mass_matrix = edge.mass_matrix
#         if coupling == :undirected
#             @warn "Reconstructing an ODEEdge affects its internal dimension."
#         return ODEEdge(f! = f!,
#                        dim = dim,
#                        coupling = coupling,
#                        mass_matrix = mass_matrix,
#                        sym = sym)
#     end
# end
#
#

"""
Allow initializing StaticEdgeFunction for Power Dynamics
"""
function StaticEdgeFunction(vertices!, edges!, graph; parallel = false)
    # For reasons I don't fully understand we have to qualify the call to
    # the constructor of StaticEdgeFunction here.
    nd_ODE_Static_mod.StaticEdgeFunction(network_dynamics(vertices!, edges!, graph, parallel = parallel))
end


end # module
