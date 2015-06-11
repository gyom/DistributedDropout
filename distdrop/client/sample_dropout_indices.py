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


def sample_dropout_indices(L_params, D_dropout_prob_pairs, L_extra_suffixes):

    # identify all the parameters that represent
    # parameters instead of parameter+suffix

    L_params_root_variables_W = []
    L_params_root_variables_b = []
    L_params_extra_variables = []

    for e in L_params:
        (layer_name, layer_number, role, param_extra) = analyze_param_name(e.name)
        if param_extra is not None:
            L_params_extra_variables.append(e)
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


    # making two consecutive layers agree
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







def build_hierarchy(list_of_params):

    # you can specify the dropout_probs as a dict indexed
    # by the layer_name (e.g. "layer_0" and "layer_17"),
    # and it should contain a tuple of two real numbers
    # like (0.0, 0.5)




    layers = set()
    shapes = {}
    dropout_probs = {}
    rough_kinds = {}
    L_extra = []

    for param_desc in list_of_params :

        (layer_name, layer_number, role, param_extra) = analyze_param_name(param_desc["name"])
        #(layer_name, layer_number, role) = analyze_param_name(param_desc["name"])
        #print "parsed (%s, %d, %s)" % (layer_name, layer_number, role)

        # sanity check. if the user supplied the layer number
        # then check that it's in agreement with the number in the name
        if param_desc.has_key("layer_number"):
            assert param_desc["layer_number"] == layer_number
        else:
            param_desc["layer_number"] = layer_number

        layers.add(param_desc["layer_number"])
        if dropout_probs.has_key(layer_name):
            assert dropout_probs[layer_name] == param_desc["dropout_probs"]
        else:
            dropout_probs[layer_name] = param_desc["dropout_probs"]

        if rough_kinds.has_key(layer_name):
            assert rough_kinds[layer_name] == proj_rough_kind(param_desc["kind"])
        else:
            rough_kinds[layer_name] = proj_rough_kind(param_desc["kind"])

        if layer_name not in shapes.keys():
            shapes[layer_name] = {}

        if param_extra not in L_extra:
            if param_extra is not None:
                L_extra.append(param_extra)

        shapes[layer_name][role] = param_desc["shape"]

    
    full_dict = {}
    full_dict["nbr_layers"] = len(layers)
    full_dict["shapes"] = shapes
    # sorta misleading because that `dropout_probs` entry is a dict of `dropout_probs` (pairs of real numbers)
    full_dict["dropout_probs"] = dropout_probs
    full_dict["rough_kinds"] = rough_kinds


    return full_dict, L_extra




# TO DO add extra to filter the parameters
def build_dropout_priv(nbr_layers, shapes, rough_kinds, dropout_probs, L_extra):
    dict_index={}

    for index in xrange(nbr_layers):
        name_layer = "layer_"+str(index)
        dict_index["layer_"+str(index)] = build_dropout(shapes[name_layer]["W"][:2],
                                                        dropout_probs[name_layer])

    # making two consecutive layers agree
    for index in xrange(1,nbr_layers):
        name_layer = "layer_"+str(index)
        name_layer_prev = "layer_"+str(index-1)

        # 1 FULLY_CONNECTED -> FULLY_CONNECTED
        if (rough_kinds[name_layer] == "FULLY_CONNECTED" and 
            rough_kinds[name_layer_prev] == "FULLY_CONNECTED"):
            dict_index[name_layer][0] = dict_index[name_layer_prev][1]

        # 2 CONV_FILTER -> CONV_FILTER
        if (rough_kinds[name_layer] == "CONV_FILTER" and 
            rough_kinds[name_layer_prev] == "CONV_FILTER"):
            dict_index[name_layer][1] = dict_index[name_layer_prev][0]

        # 3 CONV_FILTER -> FULLY_CONNECTED
        if (rough_kinds[name_layer] == "FULLY_CONNECTED" and 
            rough_kinds[name_layer_prev] == "CONV_FILTER"):

            shape_input = shapes[name_layer_prev]["W"][1]
            shape_output = shapes[name_layer]["W"][0]
            c = shape_output/shape_input
            dict_index[name_layer][0] = np.concatenate(
                                        [np.arange(index*c, (index+1)*c)
                                            for index in dict_index[name_layer_prev][1]],
                                        axis=0).astype(np.intc)

        if (rough_kinds[name_layer] == "CONV_FILTER" and 
            rough_kinds[name_layer_prev] == "FULLY_CONNECTED"):

            raise Exception("FULLY_CONNECTED ->CONV_FILTER not implemented")

    # build a dictionary for every parameters
    dict_param={}
    for index in xrange(nbr_layers):
        name_layer = "layer_"+str(index)
        name_param = "layer_"+str(index)+"_"
        dict_param[name_param+"W"] = dict_index[name_layer]

        # HERE : not just momentum find every key of the tollowing form name_param+"W_*"
        for extra in L_extra:
            dict_param[name_param+"W_"+extra] = dict_index[name_layer]

        # the original version that returned only one set of indices for biases
        #if kinds[name_layer] == "FULLY_CONNECTED":
        #    dict_param[name_param+"b"] = [dict_index[name_layer][1]]
        #    dict_param[name_param+"b_momentum"] = [dict_index[name_layer][1]]
        #else:
        #    dict_param[name_param+"b"] = [dict_index[name_layer][0]]
        #    dict_param[name_param+"b_momentum"] = [dict_index[name_layer][0]]

        just_the_zero_index = np.array([0], dtype=np.intc)
        if rough_kinds[name_layer] == "FULLY_CONNECTED":
            dict_param[name_param+"b"] = [just_the_zero_index, dict_index[name_layer][1]]
            # HERE : not just momentum find every key of the tollowing form name_param+"b_*"
            for extra in L_extra:
                dict_param[name_param+"b_"+extra] = [just_the_zero_index, dict_index[name_layer][1]]
        else:
            dict_param[name_param+"b"] = [dict_index[name_layer][0], just_the_zero_index]
            # HERE : not just momentum find every key of the tollowing form name_param+"b_*"
            for extra in L_extra:
                dict_param[name_param+"b_"+extra] = [dict_index[name_layer][0], just_the_zero_index]
        

    return dict_param

def build_dropout_index(dict_params, L_extra):

    return build_dropout_priv(nbr_layers=dict_params["nbr_layers"],
                              shapes=dict_params["shapes"],
                              rough_kinds=dict_params["rough_kinds"],
                              dropout_probs=dict_params["dropout_probs"],
                              L_extra=L_extra)


if __name__ == '__main__':

    #np.random.seed(42)

    path = "melanie_mnist_params_desc_02.json"
    D = load_params_json_from_client(path)
    dict_params, L_extra = build_hierarchy(D)
    print build_dropout_index(dict_params, L_extra)
