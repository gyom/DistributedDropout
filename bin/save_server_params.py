
import sys, os
import getopt
import pickle

import numpy as np

from distdrop.client.client_api import ClientCNNAutoSplitter

def usage():
    print "python save_server_params.py --server=127.0.0.1 --port=8200 --save_path='D_params.pkl'"


def run(server, port, save_path):

    client = ClientCNNAutoSplitter.new_basic_alpha_beta(server, port, 0.0, 1.0)

    client.connect()
    L_param_desc = client.read_param_desc_from_server()

    D_param_values = {}

    for param_desc in L_param_desc:

        name = param_desc['name']
        value = client.pull_entire_param(name)

        assert not np.any(np.isnan(value)), "The server parameter %s contains NaN values" % name

        D_param_values[name] = value

        print "Read %s of shape %s." % (name, str(param_desc['shape']))

    print "================================"

    pickle.dump(D_param_values, open(save_path, "w"))
    print "Wrote %s." % save_path

    client.quit()
    client.close()


def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["server=", "port=", "save_path="])
                                                        
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    server = "127.0.0.1"
    port = None
    save_path = None

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
        elif o in ("--save_path"):
            save_path = a
        else:
            assert False, "unhandled option"

    assert port

    run(server, port, save_path)


if __name__ == "__main__":
    main(sys.argv)

