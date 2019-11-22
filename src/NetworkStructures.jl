module NetworkStructures

using LightGraphs
using LinearAlgebra
using SparseArrays
#= This module contains the logic that calculate the index structures
and data access structs that Network Dynamics makes use of.

The key structure is the GraphData structure that allows accessing data on
vertices and edges of the graph in an efficient manner. The neccessary indices
are precomputed in GraphStructure.
=#

# We need rather complicated sets of indices into the arrays that hold the
# vertex and the edge variables. We precompute everything we can and store it
# in GraphStruct.

export create_idxs, create_offsets, GraphStruct, GraphData, EdgeData, VertexData, construct_mass_matrix

const Idx = UnitRange{Int}

"""
Create indices for stacked array of dimensions dims
"""
function create_idxs(dims; counter=1)::Array{Idx, 1}
    idxs = [1:1 for dim in dims]
    for (i, dim) in enumerate(dims)
        idxs[i] = counter:(counter + dim - 1)
        counter += dim
    end
    idxs
end

"""
Create offsets for stacked array of dimensions dims
"""
function create_offsets(dims; counter=0)::Array{Int, 1}
    offs = [1 for dim in dims]
    for (i, dim) in enumerate(dims)
        offs[i] = counter
        counter += dim
    end
    offs
end

"""
Create indexes for stacked array of dimensions dims using the offsets offs
"""
function create_idxs(offs, dims)::Array{Idx, 1}
    idxs = [1+off:off+dim for (off, dim) in zip(offs, dims)]
end

"""
This struct holds the offsets and indices for all relevant aspects of the graph
The assumption is that there will be two arrays, one for the vertex variables
and one for the edge variables.

The graph structure is encoded in the source and destination relationships s_e
and d_e. THese are arrays that hold the node that is the source/destination of
the indexed edge. Thus ``e_i = (s_e[i], d_e[i])``
"""
struct GraphStruct
    num_v::Int
    num_e::Int
    v_dims::Array{Int, 1}
    e_dims::Array{Int, 1}
    s_e::Array{Int, 1}
    d_e::Array{Int, 1}
    v_offs::Array{Int, 1}
    e_offs::Array{Int, 1}
    v_idx::Array{Idx, 1}
    e_idx::Array{Idx, 1}
    s_e_offs::Array{Int, 1}
    d_e_offs::Array{Int, 1}
    s_e_idx::Array{Idx, 1}
    d_e_idx::Array{Idx, 1}
    e_s_v_dat::Array{Array{Tuple{Int,Int}, 1}}
    e_d_v_dat::Array{Array{Tuple{Int,Int}, 1}}
end
function GraphStruct(g, v_dims, e_dims)
    num_v = nv(g)
    num_e = ne(g)

    s_e = [src(e) for e in edges(g)]
    d_e = [dst(e) for e in edges(g)]

    v_offs = create_offsets(v_dims)
    e_offs = create_offsets(e_dims)

    v_idx = create_idxs(v_offs, v_dims)
    e_idx = create_idxs(e_offs, e_dims)

    s_e_offs = [v_offs[s_e[i_e]] for i_e in 1:num_e]
    d_e_offs = [v_offs[d_e[i_e]] for i_e in 1:num_e]

    s_e_idx = [v_idx[s_e[i_e]] for i_e in 1:num_e]
    d_e_idx = [v_idx[d_e[i_e]] for i_e in 1:num_e]

    e_s_v_dat = [[(offset, dim) for (i_e, (offset, dim)) in enumerate(zip(e_offs, e_dims)) if i_v == s_e[i_e]] for i_v in 1:num_v]
    e_d_v_dat = [[(offset, dim) for (i_e, (offset, dim)) in enumerate(zip(e_offs, e_dims)) if i_v == d_e[i_e]] for i_v in 1:num_v]

    GraphStruct(
    num_v,
    num_e,
    v_dims,
    e_dims,
    s_e,
    d_e,
    v_offs,
    e_offs,
    v_idx,
    e_idx,
    s_e_offs,
    d_e_offs,
    s_e_idx,
    d_e_idx,
    e_s_v_dat,
    e_d_v_dat)
end

# In order to access the data in the arrays efficiently we create views that
# allow us to efficiently index into the underlying arrays.

import Base.getindex, Base.setindex!, Base.length


struct EdgeData{G}
    gd::G
    idx_offset::Int
    len::Int
end

@inline Base.@propagate_inbounds function getindex(e_dat::EdgeData, idx)
    e_dat.gd.e_array[idx + e_dat.idx_offset]
end

@inline Base.@propagate_inbounds function setindex!(e_dat::EdgeData, x, idx)
    e_dat.gd.e_array[idx + e_dat.idx_offset] = x
    nothing
end

@inline function length(e_dat::EdgeData)
    e_dat.len
end



struct VertexData{G}
    gd::G
    idx_offset::Int
    len::Int
end

@inline Base.@propagate_inbounds function getindex(v_dat::VertexData, idx)
    v_dat.gd.v_array[idx + v_dat.idx_offset]
end

@inline Base.@propagate_inbounds function setindex!(v_dat::VertexData, x, idx)
    v_dat.gd.v_array[idx + v_dat.idx_offset] = x
    nothing
end

@inline function length(v_dat::VertexData)
    v_dat.len
end

# Putting the above together we create a GraphData object:

# An alternative design that needs to be evaluated for performance is to create
# only one array of VertexData and EdgeData and index into that, possibly with a
# new set of access types...

mutable struct GraphData{T}
    v_array::T
    e_array::T
    v::Array{VertexData{GraphData{T}}, 1}
    e::Array{EdgeData{GraphData{T}}, 1}
    v_s_e::Array{VertexData{GraphData{T}}, 1} # the vertex that is the source of e
    v_d_e::Array{VertexData{GraphData{T}}, 1} # the vertex that is the destination of e
    e_s_v::Array{Array{EdgeData{GraphData{T}}, 1}, 1} # the edges that have v as source
    e_d_v::Array{Array{EdgeData{GraphData{T}}, 1}, 1} # the edges that have v as destination
    function GraphData{T}(v_array::T, e_array::T, gs::GraphStruct) where T
        gd = new{T}(v_array, e_array)
        gd.v = [VertexData{GraphData{T}}(gd, offset, dim) for (offset,dim) in zip(gs.v_offs, gs.v_dims)]
        gd.e = [EdgeData{GraphData{T}}(gd, offset, dim) for (offset,dim) in zip(gs.e_offs, gs.e_dims)]
        gd.v_s_e = [VertexData{GraphData{T}}(gd, offset, dim) for (offset,dim) in zip(gs.s_e_offs, gs.v_dims[gs.s_e])]
        gd.v_d_e = [VertexData{GraphData{T}}(gd, offset, dim) for (offset,dim) in zip(gs.d_e_offs, gs.v_dims[gs.d_e])]
        gd.e_s_v = [[EdgeData{GraphData{T}}(gd, offset, dim) for (offset,dim) in e_s_v] for e_s_v in gs.e_s_v_dat]
        gd.e_d_v = [[EdgeData{GraphData{T}}(gd, offset, dim) for (offset,dim) in e_d_v] for e_d_v in gs.e_d_v_dat]
        gd
    end
end

function GraphData(v_array, e_array, gs)
    GraphData{typeof(v_array)}(v_array, e_array, gs)
end




function construct_mass_matrix(mmv_array, dim_nd, gs)
    if all([mm == I for mm in mmv_array])
        mass_matrix = I
    else
        mass_matrix = sparse(1.0I,dim_nd,dim_nd)
        for (i, mm) in enumerate(mmv_array)
            if mm != I
                mass_matrix[gs.v_idx[i],gs.v_idx[i]] .= mm
            end
        end
    end
    mass_matrix
end

function construct_mass_matrix(mmv_array, mme_array, dim_v, dim_e, gs)
    if all([mm == I for mm in mmv_array]) && all([mm == I for mm in mme_array])
        mass_matrix = I
    else
        dim_nd = dim_v + dim_e
        mass_matrix = sparse(1.0I,dim_nd,dim_nd)
        for (i, mm) in enumerate(mmv_array)
            if mm != I
                mass_matrix[gs.v_idx[i],gs.v_idx[i]] .= mm
            end
        end
        for (i, mm) in enumerate(mme_array)
            if mm != I
                mass_matrix[gs.e_idx[i] + dim_v, gs.e_idx[i] + dim_v] .= mm
            end
        end
    end
    mass_matrix
end

end # module
