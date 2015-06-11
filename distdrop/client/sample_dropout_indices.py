import numpy as np
import re

def proj_rough_kind(kind):
    if kind in ["CONV_FILTER_BIASES", "CONV_FILTER_WEIGHTS"]:
        return "CONV_FILTER"
    elif kind in ["FULLY_CONNECTED_BIASES", "FULLY_CONNECTED_WEIGHTS"]:
        return "FULLY_CONNECTED"
    else:
        return None


def get_single_dropout_indices(shape, dropout_prob_pair):
    assert len(dropout_prob_pair) == 2
                                                            
    # check for cases where we want no dropout

    if dropout_prob_pair[0] == 0.0:                               
        index_row = np.arange(shape[0])
    else:
        (N, p) = (shape[0], dropout_probs[0])
        N_kept = N-int(N*p)
        index_row = np.sort(np.random.permutation(N)[:N_kept])
        assert len(index_row) == N_kept
                    
    if dropout_prob_pair[1] == 0.0:       
        index_col = np.arange(shape[1])
    else:                                                                        
        (N, p) = (shape[1], dropout_prob_pair[1])
        N_kept = N-int(N*p)
        index_col = np.sort(np.random.permutation(N)[:N_kept])
        assert len(index_col) == N_kept

    index_row = index_row.astype(np.intc)
    index_col = index_col.astype(np.intc)

    return [index_row, index_col]



def analyze_param_name(name):
    prog = re.compile(r"(layer_(\d+))_([^_]*)(_(.*)){0,1}")
    m = prog.match(name)
    if m:
        layer_name = m.group(1)
        layer_number = int(m.group(2))
        role = m.group(3)
        param_extra = m.group(5)
        return (layer_name, layer_number, role, param_extra)
    else:
        print "Failed to get layer number from %s." % name
        return None


def sample_dropout_indices(L_params, D_dropout_prob_pairs):

    # identify all the parameters that represent
    # parameters instead of parameter+suffix

    L_params_root_variables_W = []
    L_params_root_variables_b = []

    for e in L_params:
        (layer_name, layer_number, role, param_extra) = analyze_param_name(e.name)
        if param_extra is not None:
            #L_params_extra_variables.append(e)
            pass
        elif role == 'W':
            L_params_root_variables_W.append(e)
        elif role == 'b':
            L_params_root_variables_b.append(e)
        else:
            raise Exception("bug !")

    splits_for_W = {}
    rough_kinds = {}
    # splits_for_W is indexed by keys like r"layer_\d_W"
    for e in L_params_root_variables_W:
        (layer_name, _, _, _) = analyze_param_name(e.name)
        splits_for_W[e.name] = get_single_dropout_indices(e.shape, D_dropout_prob_pairs[layer_name])
        rough_kinds[layer_name] = proj_rough_kind(e["kind"])


    # sort by layer_number, just in case that wasn't already the case
    L_params_root_variables_W.sort(key=lambda e: analyze_param_name(e.name)[1])


    # making two consecutive layers agree on the splits for "W"
    for (e, e_next) in zip(L_params_root_variables_W, L_params_root_variables_W[1:]):
        
        (layer_name, _, _, _) = analyze_param_name(e.name)
        (layer_name_next, _, _, _) = analyze_param_name(e_next.name)

        # 1 FULLY_CONNECTED -> FULLY_CONNECTED
        if (rough_kinds[layer_name_next] == "FULLY_CONNECTED" and 
            rough_kinds[layer_name] == "FULLY_CONNECTED"):
            splits_for_W[layer_name_next][0] = splits_for_W[layer_name][1]

        # 2 CONV_FILTER -> CONV_FILTER
        if (rough_kinds[layer_name_next] == "CONV_FILTER" and 
            rough_kinds[layer_name] == "CONV_FILTER"):
            splits_for_W[layer_name_next][1] = splits_for_W[layer_name][0]

        # 3 CONV_FILTER -> FULLY_CONNECTED
        if (rough_kinds[layer_name_next] == "FULLY_CONNECTED" and 
            rough_kinds[layer_name] == "CONV_FILTER"):

            shape_input = e.shape[1]
            shape_output = e_next.shape[0]
            c = shape_output/shape_input
            splits_for_W[layer_name_next][0] = np.concatenate(
                                        [np.arange(index*c, (index+1)*c)
                                            for index in splits_for_W[layer_name][1]],
                                        axis=0).astype(np.intc)

        if (rough_kinds[layer_name_next] == "CONV_FILTER" and 
            rough_kinds[layer_name] == "FULLY_CONNECTED"):

            raise Exception("FULLY_CONNECTED ->CONV_FILTER not implemented")

    # assign the matching splits for "b" corresponding to the splits on "W"

    splits_for_b = {}
    just_the_zero_index = np.array([0], dtype=np.intc)
    for eb in L_param_root_variables_b:
        (layer_name, _, _, _) = analyze_param_name(eb.name)

        if rough_kinds[layer_name] == "FULLY_CONNECTED":
            splits_for_b[layer_name] = [just_the_zero_index, splits_for_W[layer_name][1]]
        elif rough_kinds[layer_name] == "CONV_FILTER":
            splits_for_b[layer_name] = [splits_for_W[layer_name][0], just_the_zero_index]]
        else:
            raise Exception("bug !")

    # and now we take care of everything by generating a list of splits
    # that's indexed by the full names (ex : "layer_0_b_momentum") instead
    # of by the layer_name (ex : "layer_0")
    splits_indices = {}
    for e in L_params:
        (layer_name, _, role, _) = analyze_param_name(e.name)
        if role == "W":
            splits_indices[e.name] = splits_for_W[layer_name]
        elif role == "b":
            splits_indices[e.name] = splits_for_b[layer_name]
        else:
            raise Exception("bug !")

    return splits_indices


