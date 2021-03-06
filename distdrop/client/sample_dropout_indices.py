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
        (N, p) = (shape[0], dropout_prob_pair[0])
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


def sample_dropout_indices(L_params_desc, D_dropout_prob_pairs):

    # identify all the parameters that represent
    # parameters instead of parameter+suffix

    L_params_desc_root_variables_W = []
    L_params_desc_root_variables_b = []

    for e in L_params_desc:
        (layer_name, layer_number, role, param_extra) = analyze_param_name(e['name'])
        if param_extra is not None:
            #L_params_desc_extra_variables.append(e)
            pass
        elif role == 'W':
            L_params_desc_root_variables_W.append(e)
        elif role == 'b':
            L_params_desc_root_variables_b.append(e)
        else:
            raise Exception("bug !")

    splits_for_W = {}
    rough_kinds = {}
    # splits_for_W is indexed by keys like r"layer_\d_W"
    for e in L_params_desc_root_variables_W:
        (layer_name, _, _, _) = analyze_param_name(e['name'])
        rough_kinds[layer_name] = proj_rough_kind(e["kind"])
        # this is because the `get_single_dropout_indices` does not
        # want to know what kind of kind of parameter it's dealing with,
        # but it needs to know which indices are the input and the output
        if rough_kinds[layer_name] == "CONV_FILTER":
            [index_col, index_row] = get_single_dropout_indices([e['shape'][1], e['shape'][0]], D_dropout_prob_pairs[layer_name])
            splits_for_W[layer_name] = [index_row, index_col]
        elif rough_kinds[layer_name] == "FULLY_CONNECTED":
            splits_for_W[layer_name] = get_single_dropout_indices([e['shape'][0], e['shape'][1]], D_dropout_prob_pairs[layer_name])
        else:
            raise Exception("bug !")


    # sort by layer_number, just in case that wasn't already the case
    L_params_desc_root_variables_W.sort(key=lambda e: analyze_param_name(e['name'])[1])

    
    #print "BEFORE HARMONIZATION."
    #for (k,v) in sorted(splits_for_W.items()):
    #    print "splits_for_W[%s] has shapes : [%s, %s]" % (k, v[0].shape, v[1].shape)


    # making two consecutive layers agree on the splits for "W"
    for (e, e_next) in zip(L_params_desc_root_variables_W, L_params_desc_root_variables_W[1:]):
        
        (layer_name, _, _, _) = analyze_param_name(e['name'])
        (layer_name_next, _, _, _) = analyze_param_name(e_next['name'])

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

            
            # Be VERY careful about this section.
            # The convolution layers have shape (OUT, IN, H, W).
            # The fully-connected layers have shape (IN, OUT, 1, 1).
            # This means that you want e['shape'][0] to match with e_next['shape'][0].
            # You are matching OUT -> IN.
            #
            # Older versions of this section had
            # #shape_input = e['shape'][1]
            # but this is not correct.
            # Also, older versions used 
            # # np.concatenate([np.arange(int(index*c), int((index+1)*c))
            #                            for index in splits_for_W[layer_name][1]],
            #                            axis=0).astype(np.intc)
            # which is also wrong.

            shape_input = e['shape'][0]
            shape_output = e_next['shape'][0]
            c = shape_output/shape_input

            if c * shape_input != shape_output:
                print "You have a problem with your configuration of dropout at the junction of the convolution and fully-connected layers."
                print "You have %d filters (%d, %d) coming out of the convolution," % (shape_input, e['shape'][2], e['shape'][3])
                print "but you then have %d units at the entrance to the fully-connected section." % shape_output
                print ""
                print "e['shape'] : %s" % str(e['shape'])
                print "e_next['shape'] : %s" % str(e_next['shape'])
                print ""
                raise Exception("Setup for split indices at CONV_FILTER -> FULLY_CONNECTED cannot proceed.")

            # Basically, here we're taking chunks of indices [0,1,2,3] and [8,9,10,11]
            # that correspond to filters of size (2,2), for example.
            # This is a bit tricky, and there are multiple ways that we could implement this.

            splits_for_W[layer_name_next][0] = np.concatenate(
                                        [np.arange(int(index*c), int((index+1)*c))
                                            for index in splits_for_W[layer_name][0]],
                                        axis=0).astype(np.intc)

            #import pdb; pdb.set_trace()

        if (rough_kinds[layer_name_next] == "CONV_FILTER" and 
            rough_kinds[layer_name] == "FULLY_CONNECTED"):

            raise Exception("FULLY_CONNECTED -> CONV_FILTER not implemented")


    #print "AFTER HARMONIZATION."
    #for (k,v) in sorted(splits_for_W.items()):
    #    print "splits_for_W[%s] has shapes : [%s, %s]" % (k, v[0].shape, v[1].shape)


    # assign the matching splits for "b" corresponding to the splits on "W"

    splits_for_b = {}
    just_the_zero_index = np.array([0], dtype=np.intc)
    for eb in L_params_desc_root_variables_b:
        (layer_name, _, _, _) = analyze_param_name(eb['name'])

        if rough_kinds[layer_name] == "FULLY_CONNECTED":
            splits_for_b[layer_name] = [just_the_zero_index, splits_for_W[layer_name][1]]
        elif rough_kinds[layer_name] == "CONV_FILTER":
            splits_for_b[layer_name] = [splits_for_W[layer_name][0], just_the_zero_index]
        else:
            raise Exception("bug !")

    # and now we take care of everything by generating a list of splits
    # that's indexed by the full names (ex : "layer_0_b_momentum") instead
    # of by the layer_name (ex : "layer_0")
    splits_indices = {}
    for e in L_params_desc:
        (layer_name, _, role, _) = analyze_param_name(e['name'])
        if role == "W":
            splits_indices[e['name']] = splits_for_W[layer_name]
        elif role == "b":
            splits_indices[e['name']] = splits_for_b[layer_name]
        else:
            raise Exception("bug !")

    return splits_indices


