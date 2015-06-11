

import numpy as np

from distdrop.client.client_api import ClientCNNAutoSplitter
from distdrop.client import messages


def mini_training(client):

    D_dropout_probs = {'layer_0' : [0.5, 0.0],
                       'layer_1' : [0.5, 0.5], 
                       'layer_2' : [0.5, 0.5], 
                       'layer_3' : [0.5, 0.0]}

    # maybe call client.push_entire_param to initialize the paramater from your favorite pickle file
    # (with only ONE client at the beginning of everything !)

    # while (not done):
    #     #this is for one mini-batch
    #
    #     client.perform_split(D_dropout_probs)
    #
    #
    #     for each of your parameters that you need
    #         param_value = client.pull_split_param(name)
    #         store param_value in your theano model
    #
    #     run your training algorithm for a few mini-batches
    #
    #     for each of your parameters that you need
    #         param_value = get the value in your theano model
    #         client.push_split_param(name, value) to communicate the model back to the server
    #
    # onwards to the next mini-batch !

    # maybe call client.pull_entire_param to read everything and save it to your favorite pickle file
    # (don't overwrite the same file with all the clients !)



def test_slice(client):

    # based on "melanie_mnist_params_desc_02.json"

    D_dropout_probs = {'layer_0' : [0.5, 0.0],
                       'layer_1' : [0.5, 0.5], 
                       'layer_2' : [0.5, 0.5], 
                       'layer_3' : [0.5, 0.0]}

    # name = "layer_3_W"
    name = "layer_0_W"

    # initialize
    client.push_entire_param(name, np.arange(0, 1000, dtype=np.intc).reshape((40, 1, 5, 5)), 0.0, 1.0)
    
    client.perform_split(D_dropout_probs)
    
    #print client.splits_indices["layer_0_W"]
    #print "the sub parameter contains"
    A = client.pull_split_param(name)
    assert A.shape == (20,1,5,5)

    B = np.random.rand(*A.shape)
    #B = np.ones(A.shape)

    client.push_split_param(name, B)

    C = client.pull_split_param(name)

    Cref = client.alpha*A + client.beta*B
    diff = np.max(np.abs(C-Cref))
    print "Maximum abs difference is %f." % diff


def test_slice2(client):

    # based on "melanie_mnist_params_desc_02.json"
    #D_dropout_probs = {'layer_0_W' : [0.5, 0.0], "layer_0_W_momentum" : [0.5, 0.0], "layer_0_b" : [0.5, 0.0], "layer_0_b_momentum" : [0.5, 0.0],
    #                   'layer_1_W' : [0.5, 0.5], "layer_1_W_momentum" : [0.5, 0.5], "layer_1_b" : [0.5, 0.5], "layer_1_b_momentum" : [0.5, 0.5],
    #                   'layer_2_W' : [0.5, 0.5], "layer_2_W_momentum" : [0.5, 0.5], "layer_2_b" : [0.5, 0.5], "layer_2_b_momentum" : [0.5, 0.5],
    #                   'layer_3_W' : [0.5, 0.0], "layer_3_W_momentum" : [0.5, 0.0], "layer_3_b" : [0.5, 0.0], "layer_3_b_momentum" : [0.5, 0.0]}

    D_dropout_probs = {'layer_0' : [0.5, 0.0],
                       'layer_1' : [0.5, 0.5], 
                       'layer_2' : [0.5, 0.5], 
                       'layer_3' : [0.5, 0.0]}

    # name = "layer_3_W"
    name = "layer_0_W"

    # initialize
    client.push_entire_param(name, np.arange(0, 1000, dtype=np.intc).reshape((40, 1, 5, 5)), 0.0, 1.0)

    print "the full parameter contains"
    F = client.pull_entire_param(name)
    print F[:,0,0,0].flatten()
    #print F[:,:,0,0]


    
    client.perform_split(D_dropout_probs)
    
    # manual crafting of the splits to debug.
    # messes up one of the splits
    client.splits_indices[name] = (np.array([0,2,4,6], dtype=np.intc), np.array([0], dtype=np.intc))


    #print client.splits_indices["layer_0_W"]
    print "the sub parameter contains"
    A = client.pull_split_param(name)
    #assert A.shape == (20,1,5,5)

    #print A[:,:,0,0]

    print A[:,0,0,0].flatten()


    #B = np.random.rand(*A.shape)
    B = np.ones(A.shape)

    client.push_split_param(name, B)

    C = client.pull_split_param(name)

    print "the sub parameter contains"
    print C[:,0,0,0].flatten()
    #print C[:,:,0,0]


    print "the full parameter contains"
    F = client.pull_entire_param(name)
    print F[:,0,0,0].flatten()
    #print F[:,:,0,0]

    #Cref = client.alpha*A + client.beta*B
    #diff = np.max(np.abs(C-Cref))
    #print "Maximum abs difference is %f." % diff


def run():

    server_host = "127.0.0.1"
    port = 6000

    (alpha, beta) = (0.0, 1.0)
    client = ClientCNNAutoSplitter.new_basic_alpha_beta(server_host, port, alpha, beta)

    client.connect()
    print client.read_param_desc_from_server()
    client.save_all_to_hdf5("test0")
    client.load_all_from_hdf5("test0JSON", "test0")
    # mini_training()
    
    test_slice(client)

    client.quit()
    client.close()


if __name__ == "__main__":
    run()







