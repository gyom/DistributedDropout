
import sys, os
import getopt
import time


import numpy as np
import numpy.testing as npt

from client_api import ClientCNNAutoSplitter

def usage():
    print "python test_soak_server_split_params.py --server=127.0.0.1 --port=5000 --nclients=5 --nreps=1 --nsplits=2"


def extract_layer_names(L_params):
    res = set([])
    #prog = re.compile(r"(.+_.+)_.*")
    for param in L_params:
        (layer_name, layer_number, role) = ClientCNNAutoSplitter.analyze_param_name(param['name'])
        res.add(layer_name)
        #m = prog.match(param['name'])
        #if m:
        #    res.add(m.group(1))
        #else:
        #    print "Failed to extract layer name from %s" % name
    return res



def run(server, port, nclients, nreps, nsplits):

    # we're not doing the clients in parallel or it'll be an insane mess


    for _ in range(nclients):

        (alpha, beta) = (np.random.rand(), np.random.rand())
        print "Starting with new client. (alpha, beta) are (%0.2f, %0.2f)" % (alpha, beta)
        client = ClientCNNAutoSplitter.new_basic_alpha_beta(server, port, alpha, beta)

        client.connect()

        print "    Starting soak."
        for __ in range(nreps):
            soak(client, alpha, beta, nsplits)
        print "   Done with soak."

        client.quit()
        client.close()
        print "Done with client."


def soak(client, alpha, beta, nsplits):

    L_params = client.read_param_desc_from_server()
    layer_names = extract_layer_names(L_params)
    # TODO : Consider other dropout ratios.
    D_dropout_probs = dict((ln, [0.5, 0.5]) for ln in layer_names)

    current_param_entire_values = {}

    for _ in range(nsplits):

        print "        Reading the current values."
        np.random.shuffle(L_params)
        for param in L_params:
            name = param['name']
            current_param_entire_values[name] = client.pull_entire_param(name)

        print "        New split proposed."
        client.perform_split(D_dropout_probs)

        np.random.shuffle(L_params)
        for param in L_params:
            name = param['name']
            v_current = client.pull_split_param(name)

            splits_indices = client.splits_indices[name]
            E = current_param_entire_values[name]
            subE = E[splits_indices[0],:,:,:][:,splits_indices[1],:,:]

            npt.assert_allclose(v_current, subE, atol=1e-8)
            del subE

            v_update = np.random.randn(*v_current.shape).astype(np.float32)
            client.push_split_param(name, v_update)

            v_resulting = client.pull_split_param(name)

            npt.assert_allclose(v_resulting, alpha*v_current + beta*v_update, atol=1e-8)
            
            print "            Ran through %s." % name

        print "        Done with this split."




def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["server=", "port=",
                                                        "nclients=", "nreps=", "nsplits="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    server = "127.0.0.1"
    port = None
    nclients = 5
    nreps = 1
    nsplits = 2

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
        elif o in ("--nclients"):
            nclients = int(a)
        elif o in ("--nreps"):
            nreps = int(a)
        elif o in ("--nsplits"):
            nsplits = int(a)
        else:
            assert False, "unhandled option"

    assert port

    run(server, port, nclients, nreps, nsplits)


if __name__ == "__main__":
    main(sys.argv)

