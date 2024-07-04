function check_euler(fs)
    return sum(f * (-1)^i for (i, f) in enumerate(fs))
end

function composition_polyhedron(M)
    half_spaces = [HalfSpace(M[i, :], 0) for i in axes(M, 1)]
    H = reduce(∩, half_spaces)
    return polyhedron(H, CDDLib.Library())
end

function singleton_sets(M; verbose=false)
    nstr, d = size(M)
    p = composition_polyhedron(M)

    if verbose
        println("Removing redudancy...")
    end

    removehredundancy!(p)

    if verbose
        println("done!")
    end

    #TODO: check for scalar multiples
    K = convert(Matrix{Int}, MixedMatHRep(p).A)
    designable_strs = zeros(Int, size(K, 1))
    for (s, m) in enumerate(eachrow(M))
        for (i, k) in enumerate(eachrow(K))
            if all(m .== k)
                designable_strs[i] = s
            end
        end
    end
    return designable_strs
end

function rays(M; verbose=true)
    p = composition_polyhedron(M)

    if verbose
        println("Removing redudancy...")
    end

    removehredundancy!(p)

    if verbose
        println("done!")
    end

    v = vrep(p)
    v = MixedMatVRep(v)
    R = permutedims(v.R)
    R ./= sqrt.(sum(R .^ 2, dims=1))
    return R
end

function count_faces(M; verbose=true, compute_diagram=false, thresh=1e-10)
    n_structs, d = size(M)

    R = rays(M; verbose=verbose)
    n_rays = size(R, 2)
    incidences = abs.(M * R) .< thresh

    fs = [1; n_rays; zeros(Int64, d + 1 - 2)]
    sets = BitMatrix[trues(n_structs, 1), incidences]

    k = 2
    while fs[k] > 1
        if verbose
            println("Finding faces of dimension $k...")
        end

        new_set = Set()
        max_idx = size(sets[k], 2)

        removed_indices = []
        for i in 1:max_idx, j in (i + 1):max_idx
            x, y = sets[k][:, i], sets[k][:, j]
            pair = x .& y

            if pair ∈ new_set
                continue
            end

            if pair == x
                push!(removed_indices, i)

                push!(new_set, pair)
                fs[k + 1] += 1
            elseif pair == y
                push!(removed_indices, j)

                push!(new_set, pair)
                fs[k + 1] += 1

            elseif pair ∉ eachcol(sets[k])
                push!(new_set, pair)
                fs[k + 1] += 1
            end
        end

        sets[k] = sets[k][:, setdiff(1:end, removed_indices)]
        fs[k] -= length(removed_indices)

        # TODO CLEAN THIS UP
        if length(new_set) > 1
            new_s = reduce(hcat, new_set)
        else
            new_s = BitMatrix(undef, n_structs, 1)
            new_s[:, 1] .= only(new_set)
        end
        push!(sets, new_s)
        k += 1
    end

    if verbose
        println("done!")
    end

    if !compute_diagram
        return fs, sets
    end

    diagram = DiGraph(sum(fs))
    node_labels = Dict{Int,Tuple{Int,Int}}()
    node_labels[1] = (0, 1)
    set_nodes = Dict{Tuple{Int,Int},Int}()
    set_nodes[0, 1] = 1


    for j in 2:sum(fs[1:2])
        add_edge!(diagram, 1, j)
        node_labels[j] = (1, j-1)
        set_nodes[1, j-1] = j
    end

    for k in 2:d
        i_start = sum(fs[1:k])
        j_start = sum(fs[1:(k - 1)])

        for (i, x) in enumerate(eachcol(sets[k + 1]))
            i = i_start + i
            for (j, y) in enumerate(eachcol(sets[k]))
                j = j_start + j
                if x .& y == x
                    add_edge!(diagram, j, i)
                    node_labels[i] = (k, i - i_start)
                    set_nodes[k, i - i_start] = i
                end
            end
        end
    end
    return fs, sets, diagram, node_labels, set_nodes
end

# function count_faces(p0, d; verbose=true)
#     n_structs, f0 = size(p0)

#     fs = [1; f0; zeros(Int64, d + 1 - 2)]
#     sets = Any[ones(Bool, (n_structs, 1)), p0]

#     k = 2
#     while fs[k] > 1
#         if verbose
#             println("Finding faces of dimension $(k+1)...")
#         end

#         new_set = Set()
#         max_idx = size(sets[k], 2)

#         removed_indices = []
#         for i in 1:max_idx, j in (i + 1):max_idx
#             x, y = sets[k][:, i], sets[k][:, j]
#             pair = x .& y

#             if pair ∉ new_set
#                 if pair == x
#                     push!(removed_indices, i)

#                     push!(new_set, pair)
#                     fs[k + 1] += 1

#                 elseif pair == y
#                     push!(removed_indices, j)

#                     push!(new_set, pair)
#                     fs[k + 1] += 1

#                 elseif pair ∉ eachcol(sets[k])
#                     push!(new_set, pair)
#                     fs[k + 1] += 1
#                 end
#             end
#         end

#         sets[k] = sets[k][:, setdiff(1:end, removed_indices)]
#         fs[k] -= length(removed_indices)

#         push!(sets, reduce(hcat, new_set))
#         k += 1
#     end

#     diagram = DiGraph(sum(fs))
#     for j in 2:sum(fs[1:2])
#         add_edge!(diagram, 1, j)
#     end

#     for k in 2:d
#         i_start = sum(fs[1:k])
#         j_start = sum(fs[1:(k - 1)])

#         for (i, x) in enumerate(eachcol(sets[k + 1]))
#             i = i_start + i
#             for (j, y) in enumerate(eachcol(sets[k]))
#                 j = j_start + j
#                 if x .& y == x
#                     add_edge!(diagram, j, i)
#                 end
#             end
#         end
#     end
#     return fs, sets, diagram
# end