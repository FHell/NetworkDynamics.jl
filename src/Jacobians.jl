"""
    This module contains the components for solvinf dynamical systems in which implicit
    solvers (mostly used for stiff differential equations) are used in combination with
    jacobians and jacobian vector products (jvp).

    The key structure `NDJacVecOperator` is based on the AbstractDiffEqOperator interface
    from DifferentialEquations.jl. By using an AbstractDiffEqOperator it is possible for the
    solver to exploit linearity and hence achieve maximal performance.
    `NDJacVecOperator` contains the JacGraphData structure that allows accessing data of the
    jacobians and the jvp of the vertices and edges.
    The vector of the jvp is given by `x` which corresponds to the current vertex values
    and is updated every time step using the function update_coefficients!.
"""
module Jacobians

using ..NetworkStructures
using ..Utilities
using ..nd_ODE_Static_mod

using Reexport
using LinearAlgebra
using DiffEqBase
import DiffEqBase.update_coefficients!
export JacGraphData, NDJacVecOperator

"""
    The JacGraphData is similar to the GraphData. It contains the jacobians of the ith/jth
    vertex/edge and provides access to them. Note that the e_jac_array has an additional
    dimension since the jacobians for the outgoing and incoming edges must be considered.
    Another array (e_jac_product) is needed which stores the product of the edge jacobian
    and the vector.
    The data for specific jacobians of vertices/edges can be accessed using the
        get_vertex_jacobian, get_src_edge_jacobian, get_dst_edge_jacobian
    methods.
    For the later summation of the jvp (see for loops in jacvecprod/jacvecprod!),
    the dimension of the edge must be equal to the dimension of the destination node.
    This means that for the dimension of the e_jac_array (or for its subarrays) must hold:
        src_edge_jacobian_dim = v_src_dim x v_dst_dim
        (derivative acc. to the source node),
        dst_edge_jacobian_dim = v_dst_dim x v_dst_dim
        (derivative acc. to the destination node).
    Since we currently consider only homogeneous networks, the following is valid:
        v_src_dim = v_dst_dim = v_dim.
"""

struct JacGraphData
    v_jac_array::Array{Array{Float64, 2}, 1} # contains the jacobians for each vertex
    e_jac_array::Array{Array{Array{Float64, 2}, 1}, 1} # contains the jacobians for each edge
    e_jac_product::Array{Array{Float64, 1}, 1} # is needed later in jac_vec_prod(!) as a storage for the products of edge jacobians and vectors z
end

function JacGraphData(v_jac_array, e_jac_array, e_jac_product_array, gs::GraphStruct)
    v_jac_array = [Array{Float64,2}(undef, dim, dim) for dim in gs.v_dims]
    e_jac_array = [[zeros(dim, srcdim), zeros(dim, dstdim)] for (dim, srcdim, dstdim) in zip(gs.v_dims, gs.v_dims, gs.v_dims)] # homogene Netzwerke: v_src_dim = v_dst_dim = v_dim
    e_jac_product = [zeros(gs.v_dims[1]) for i in 1:gs.num_e]
    JacGraphData(v_jac_array, e_jac_array, e_jac_product)
end

"""
    These functions create a view-like acess to the vertex jacobians of the ith node
    and to the src_edge_jacobian/dst_edge_jacobian of the jth edges, respectively.
"""

@inline @Base.propagate_inbounds get_src_edge_jacobian(jgd::JacGraphData, i::Int) = jgd.e_jac_array[i][1]
@inline @Base.propagate_inbounds get_dst_edge_jacobian(jgd::JacGraphData, i::Int) = jgd.e_jac_array[i][2]
@inline @Base.propagate_inbounds get_vertex_jacobian(jgd::JacGraphData, i::Int) = jgd.v_jac_array[i]

"""
    NDJacVecOperator(x, p, t, vertices!, edges!, graph_structure, graph_data, jac_graph_data, parallel)

The structure `NDJacVecOperator` is based on the JacVecOperator from DiffEqOperators.jl.
The corresponding object forms the Jacobian of the differential equations with respect
to the state variable `x` at time `t` with parameters `p` and has the signature
    (J, x, p, t).
Thus, the functions for the jacobians of vertices and edges must have the following form:
    vertex_jacobian!(J, v, p, t)
    edge_jacobian!(J_s, J_d, v_s, v_d, p, t).
Note that the edge_jacobian is composed of the subarrays J_s and J_d as described above.

Further fields are an array of VertexFunctions `vertices!`,
an array of EdgeFunctions `edges!`, the structure `graph_structure` and the data `graph_data`
of the underlying graph `g` as well as the `jac_graph_data` which comprises the jacobians of
edges and jacobians. The optional argument `parallel` is a boolean
value that denotes if the central loop should be executed in parallel with the number of
threads set by the environment variable `JULIA_NUM_THREADS`.
"""

mutable struct NDJacVecOperator{T, uType, tType, T1, T2, GD, JGD} <: DiffEqBase.AbstractDiffEqLinearOperator{T} # mutable da x, p, t geupdated werden
    x::uType
    p
    t::tType
    vertices!::T1
    edges!::T2
    graph_structure::GraphStruct
    graph_data::GD
    jac_graph_data::JGD
    parallel::Bool

    function NDJacVecOperator{T}(x, p, t, vertices!, edges!, graph_structure, graph_data, jac_graph_data, parallel) where T
        new{T, typeof(x), typeof(t), typeof(vertices!), typeof(edges!), typeof(graph_data), typeof(jac_graph_data)}(x, p, t, vertices!, edges!, graph_structure, graph_data, jac_graph_data, parallel)
    end

    function NDJacVecOperator(x, p, t, vertices!, edges!, graph_structure, graph_data, jac_graph_data, parallel)
        NDJacVecOperator{eltype(x)}(x, p, t, vertices!, edges!, graph_structure, graph_data, jac_graph_data, parallel)
    end
end

"""
    update_coefficients!(J, x, p, t)

updates the Jacobians of the dynamical equations (vertex_jacobian!, edge_jacobian!)
as well as the arguments `x`, `p`, `t` for every time step.
"""

function update_coefficients!(Jac::NDJacVecOperator, x, p, t)

    gs = Jac.graph_structure
    checkbounds_p(p, gs.num_v, gs.num_e)
    gd = prep_gd(x, x, Jac.graph_data, Jac.graph_structure)
    jgd = Jac.jac_graph_data

    @inbounds for i in 1:gs.num_v
        maybe_idx(Jac.vertices!, i).vertex_jacobian!(
          get_vertex_jacobian(jgd, i),
          get_vertex(gd, i),
          p_v_idx(p, i),
          t)
    end

    @inbounds for i in 1:gs.num_e
          maybe_idx(Jac.edges!, i).edge_jacobian!(
              get_src_edge_jacobian(jgd, i),
              get_dst_edge_jacobian(jgd, i),
              get_src_vertex(gd, i),
              get_dst_vertex(gd, i),
              p_e_idx(p, i),
              t)
      end

    Jac.x = x
    Jac.p = p
    Jac.t = t
end

# functions for NDJacVecOperator: both syntaxes must be taken into account: Jac, z and dx, Jac, z
"""
    Two methods are provided for the jacobian vector product.
    For the use of a differential equation of the form (x, p, t) the jac_vec_prod function
    is used which later represents the overloaded *-operator.
    For the form (dx, x, p, t), the function jac_vec_prod! is used in the function mul!.
"""

"""
    jac_vec_prod(J, z)

First, the product of the jacobians of the edges and an abstract vector `z` is calculated.
Since there is no `dx` array in this case, this needs to be prepared to store there the
product of vertex-jacobian and `z` and to sum up the corresponding entries
of the `e_jac_product` on `dx`.
"""
function jac_vec_prod(Jac::NDJacVecOperator, z)

    gs = Jac.graph_structure
    p = Jac.p
    #x = Jac.x
    checkbounds_p(p, gs.num_v, gs.num_e)
    jgd = Jac.jac_graph_data

    # first for loop that considers the mutliplication of each edge jacobians with the corresponding component of z
    for i in 1:gs.num_e
        jgd.e_jac_product[i] .= get_src_edge_jacobian(jgd, i) * view(z, gs.s_e_idx[i]) + get_dst_edge_jacobian(jgd, i) * view(z, gs.d_e_idx[i])
    end

    # in this function there is no dx in which the Jacobian can be stored, so an extra array must be created and returned
    dx = zeros(gs.v_dims[1], gs.num_v)

    # second for loop in which the multiplication of vertex jacobian and the corresponding component of z is done with addition of the e_jac_product to dx
    for i in 1:gs.num_v
        # we can use something like
        # v_cache = zeros(dim_v)
        # mul!(v_cache,  jgd.v_jac_array[i], view(z, gs.v_idx[i]))
        # however sum allocated when a vertex has multiple incoming edges
        if !isempty(gs.d_v[i])
            view(dx, gs.v_idx[i]) .= get_vertex_jacobian(jgd, i) * view(z, gs.v_idx[i]) .+ sum(view(jgd.e_jac_product, gs.d_v[i]))
        end
    end
    return dx
end

"""
    jac_vec_prod!(dx, J, z)

Analogous to function jac_vec_prod, using the already existing `dx` array.
"""

function jac_vec_prod!(dx, Jac::NDJacVecOperator, z)
    gs = Jac.graph_structure
    p = Jac.p
    #x = Jac.x
    checkbounds_p(p, gs.num_v, gs.num_e)
    jgd = Jac.jac_graph_data

    @inbounds for i in 1:gs.num_e
        # Store Edge_jac_src * v_src
        mul!(jgd.e_jac_product[i], get_src_edge_jacobian(jgd, i), view(z, gs.s_e_idx[i]))
        # in-place Add Edge_jac_dst * v_dst
        # mul!(C,A,B,α,β) = A B α + C β
        mul!(jgd.e_jac_product[i],
             get_dst_edge_jacobian(jgd, i),
             view(z, gs.d_e_idx[i]), 1, 1) # α = 1, β = 1


        #jgd.e_jac_product[i] .= get_src_edge_jacobian(jgd, i) * view(z, gs.s_e_idx[i]) .+ get_dst_edge_jacobian(jgd, i) * view(z, gs.d_e_idx[i])
    end

    @inbounds for i in 1:gs.num_v
        # Vertex Jac * vertex Variables
        mul!(view(dx, gs.v_idx[i]), get_vertex_jacobian(jgd, i), view(z, gs.v_idx[i]))

        @inbounds for j in gs.d_v[i]
            # add pre-compute edge_product (if gs.d_v not empty)
            view(dx, gs.v_idx[i]) .+= jgd.e_jac_product[j]
        end

    end
    nothing
end

# functions for NDJacVecOperator callable structs at the end of this module

Base.:*(Jac::NDJacVecOperator, z::AbstractVector) = jac_vec_prod(Jac, z)

function LinearAlgebra.mul!(dx::AbstractVector, Jac::NDJacVecOperator, z::AbstractVector)
    jac_vec_prod!(dx, Jac, z)
end

"""
    The callable structs of the NDJacVecOperator first call the `update_coefficients!`
    function to change the internal coefficents to then build the corresponding
    jacobian-vector-products using the overloaded *-operator:
    *(J, z)
    or the function:
    mul!(dx, J, z), respectively.
"""

function (Jac::NDJacVecOperator)(x, p, t) # auch Number bei t? # weglassen?
    update_coefficients!(Jac, x, p, t)
    Jac*x
end

function (Jac::NDJacVecOperator)(dx, x, p, t::Number)
    update_coefficients!(Jac, x, p, t)
    mul!(dx, Jac, x)
end


### More functions to fullfill specification

Base.size(L::NDJacVecOperator) = (length(L.x),length(L.x))
Base.size(L::NDJacVecOperator,i::Int) = length(L.x)

end # module