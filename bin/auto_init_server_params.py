
import sys, os
import getopt
import time
import re

import numpy as np

from client_api import ClientCNNAutoSplitter

def usage():
    print "python auto_init_server_params.py --server=127.0.0.1 --port=8200 --W_range=0.1 --b_range=0.1 --want_zero_momentum --use_fanin"


def run(server, port, W_range, b_range, want_zero_momentum, use_fanin):

    client = ClientCNNAutoSplitter.new_basic_alpha_beta(server, port, 0.0, 1.0)

    client.connect()
    L_params = client.read_param_desc_from_server()

    # and then you proceed to reset all the parameters to some
    # random initial values

    total_time_push = 0.0
    total_time_pull = 0.0

    total_bytes_pull = 0

    for param in L_params:

        current_time_push = 0.0
        current_time_pull = 0.0

        name = param['name']

        if param['kind'] in ['CONV_FILTER_WEIGHTS', 'FULLY_CONNECTED_WEIGHTS']:
            scale = W_range
        elif param['kind'] in ['CONV_FILTER_BIASES', 'FULLY_CONNECTED_BIASES']:
            scale = b_range
        else:
            raise Exception("You got a parameter %s with an invalid kind : %s" % (name, param['kind']))

        if re.match(r".*momentum", name) and want_zero_momentum:
            #print "momentum detected %s and want zero momentum" % name
            scale = 0.0
        
        shape = param['shape']
        assert len(shape) == 4



        if use_fanin and param['kind'] in ['CONV_FILTER_WEIGHTS', 'FULLY_CONNECTED_WEIGHTS']:

            # in this particular case, we override only if we're dealing with a weight (not a bias)
            # and we use a normal distribution instead of a uniform(-1,1)

            if param['kind'] == 'CONV_FILTER_WEIGHTS':
                std = np.sqrt(2.0/shape[1])
            elif param['kind'] == 'FULLY_CONNECTED_WEIGHTS':
                std = np.sqrt(2.0/shape[0])
            else:
                raise Exception("BUG !? % s" % param['kind'])

            updated_value = (scale * std * (np.random.randn(*shape))).astype(np.float32)

        else:
            updated_value = (scale * (np.random.rand(*shape)*2.0 - 1.0)).astype(np.float32)



        tic = time.time()
        client.push_entire_param(name, updated_value, 0.0, 1.0)
        toc = time.time()
        current_time_push = current_time_push + toc - tic

        #print "updating %s" % name
        #time.sleep(1)

        tic = time.time()
        read_value = client.pull_entire_param(name)
        toc = time.time()
        current_time_pull = current_time_pull + toc - tic

        assert not np.any(np.isnan(read_value)), "The server parameter %s contains NaN values. Unfortunately, it won't recognize your attempt at updating its value with (alpha=0.0,beta=1.0) and you will still have NaN after the arithmetic operation on NaNs." % name

        assert np.max(np.abs(read_value - updated_value)) < 1e-8, "This is a very bad sign, but it can also happen purely because of a race condition so it's not necessarily bad. Re-run the command again."

        current_bytes_pull = 4 * np.prod(updated_value.shape)

        total_time_push = total_time_push + current_time_push
        total_time_pull = total_time_pull + current_time_pull

        total_bytes_pull = total_bytes_pull + current_bytes_pull

        print "param %s updated. %d bytes, %d msec push, %d msec pull" % (name,
                                                                          current_bytes_pull,
                                                                          int(current_time_push * 1000),
                                                                          int(current_time_pull * 1000))

    print "================================"
    print ""
    print "total time spent : %d msec push, %d msec pull" % (int(total_time_push * 1000),
                                                             int(total_time_pull * 1000))
    print ""
    print "average %0.2f MB/s pull" % ((1.0 * total_bytes_pull / 1000 / 1000) / total_time_pull,)
    print ""

    client.quit()
    client.close()


def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["server=", "port=",
                                                        "W_range=", "b_range=",
                                                        "want_zero_momentum", "use_fanin"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    server = "127.0.0.1"
    port = None
    W_range = 1.0
    b_range = 1.0
    want_zero_momentum = False
    use_fanin = False

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
        elif o in ("--W_range"):
            W_range = float(a)
        elif o in ("--b_range"):
            b_range = float(a)
        elif o in ("--want_zero_momentum"):
            want_zero_momentum = True
        elif o in ("--use_fanin"):
            use_fanin = True
        else:
            assert False, "unhandled option"

    assert port

    run(server, port, W_range, b_range, want_zero_momentum, use_fanin)


if __name__ == "__main__":
    main(sys.argv)

