
import sys, os
import getopt
import pickle
import sets

import numpy as np

from distdrop.client.client_api import ClientCNNAutoSplitter

def usage():
    print "python load_server_params.py --server=127.0.0.1 --port=8200 --load_path='D_params.pkl'"


def run(server, port, load_path):

    client = ClientCNNAutoSplitter.new_basic_alpha_beta(server, port, 0.0, 1.0)

    client.connect()
    L_param_desc = client.read_param_desc_from_server()

    # late on we might want to add support for hdf5,
    # but currently we'll work only with pickle files
    D_params = pickle.load(open(load_path, "r"))
    print "Loaded %s." % load_path

    # Before we commit stuff to the server, we'll make pretty damn sure
    # that we're talking about the same thing.
    A = set(D_params.keys())
    B = set(e['name'] for e in L_param_desc)
    for e in A - B:
        print "D_params has parameter %s that the server does not have." % e
    for e in B - A:
        print "The server has parameter %s that D_params does not have." % e
    assert A == B


    for param_desc in L_param_desc:

        name = param_desc['name']
        shape = param_desc['shape']
        assert D_params.has_key(name), "Server has parameter %s but the load_path %s does not contain a value for this parameter." % (name, load_path)
        assert tuple(D_params[name].shape) == tuple(shape), "Loaded parameter %s from file. It has shape %s, but the server says that it should have shape %s." % (name, D_params[name].shape, shape)

        assert not np.any(np.isnan(D_params[name])), "The saved parameter %s contains NaN values." % name

        client.push_entire_param(name, D_params[name], 0.0, 1.0)

        print "Pushed %s of shape %s." % (name, str(param_desc['shape']))

    print "================================"

    client.quit()
    client.close()


def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["server=", "port=", "load_path="])
                                                        
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    server = "127.0.0.1"
    port = None
    load_path = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--server"):
            server = a
        elif o in ("--port"):
            port = int(a)
        elif o in ("--load_path"):
            load_path = a
        else:
            assert False, "unhandled option"

    assert port

    run(server, port, load_path)


if __name__ == "__main__":
    main(sys.argv)

