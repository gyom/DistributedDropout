
import socket
import time
import re

#from distdrop.client.client_api import ClientCNNAutoSplitter
from distdrop.client.messages import *

from distdrop.client.sample_dropout_indices import sample_dropout_indices

class Client(object):

    def __init__(self, server_host, port):
        super(Client, self).__init__()

        self.port = port
        self.server_host = server_host
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
         # `self.param_desc` will be a list when populated.
         #  its entries will be dict with keys "name", "shape", "kind".
        self.param_desc = None

    def connect(self):
        self.conn.connect((self.server_host, self.port))

    def close(self):
        self.conn.close()

    def quit(self):
        # call this before calling `close`
        header = MsgHeader('MSG_TYPE_CLIENT_QUITS')
        header.send(self.conn)

    def read_param_desc_from_server(self):
        header = MsgHeader('MSG_TYPE_LIST_ALL_PARAMS_DESC')
        header.send(self.conn)

        msg = MsgListAllParamsDesc()
        resp = msg.read_decode_response(self.conn)

        # populate self.L_param_desc to help with a lot of things
        self.L_param_desc = resp

        # for the moment we can debug a lot of things
        # by just returning that structure
        return resp
        
    def get_param_slice_from_server(self, name, S, D, indices, dtype_for_client):

        header = MsgHeader('MSG_TYPE_PULL_PARAM')
        msg = MsgPullParams(name, S, D, indices, dtype_for_client)

        header.send(self.conn)
        msg.send(self.conn)
        return msg.read_decode_response(self.conn)


    def update_param_slice_to_server(self, value, alpha, beta, name, S, D, indices, dtype_for_client):
        # value contains a numpy array
        header = MsgHeader('MSG_TYPE_PUSH_PARAM')
        msg = MsgPushParams(value, alpha, beta, name, S, D, indices, dtype_for_client)

        header.send(self.conn)
        msg.send(self.conn)
        return msg.read_decode_response(self.conn)

    def get_param_desc(self, name):
        if self.L_param_desc is None:
            self.read_param_desc_from_server()

        for param_desc in self.L_param_desc:
            if name == param_desc['name']:
                return param_desc
        

    def save_all_to_hdf5(self, path):
        header = MsgHeader('MSG_TYPE_SAVE_ALL_TO_HDF5')
        msg = MsgSaveAllToHDF5(path)
        header.send(self.conn)
        msg.send(self.conn)
        print("SENDING STUFF TO SERVER")
        return True

    def load_all_from_hdf5(self, pathHDF5, pathJSON):
        header = MsgHeader('MSG_TYPE_LOAD_ALL_FROM_HDF5')
        msg = MsgLoadAllFromHDF5(pathJSON, pathHDF5)
        header.send(self.conn)
        msg.send(self.conn)
        print("RECEIVING STUFF FROM SERVER")
        return True

# TODO : think about the possibility of having float16 
#        be used to cache the values for `want_delta_updates`

# TODO : You might want to shuffle that the list of parameters in order
#        to desynchronize the clients and speed up certain
#        operations with mutex.




# This is the class that people should be using.
# They should refrain from using the more "manual"
# parent class `Client`.

class ClientCNNAutoSplitter(Client):

    # Users are not expected to call this to construct
    # a new instance. They should use the class methods.
    # Not all of those parameters are required, and only
    # certain configurations make sense.
    def __init__(self,  server_host, port,
                        alpha, beta,
                        want_delta_updates):

        super(ClientCNNAutoSplitter, self).__init__(server_host, port)

        # indexed by root_name, just like the splits themselves.
        # The timestamps are for the duchi scaling (no longer supported).
        self.splits_indices = {}

        self.splits_timestamp_pull = {}
        self.splits_timestamp_push = {}
        self.total_time_spent_pulling = 0.0
        self.total_time_spent_pushing = 0.0

        # indexed by param name, optional.
        # Only used when `want_delta_updates` is True.
        # Consumes a lot more memory because we cache the parameters.
        self.splits_cached_values = {}

        # Just leave this there for now. Maybe we'll change that dtype_for_client
        # eventually to work with something other than float32
        # self.dtype_for_client = DTYPE_FLOAT32

        # An ordered list of the layers by number.
        # Populated when you can `read_param_desc_from_server()`
        # and assumes a very strong naming pattern with all the
        # layers named "layer_NUMBER_ROLE".
        # self.L_layer_names = []

        self.alpha = alpha
        self.beta = beta
        self.want_delta_updates = want_delta_updates        


    @classmethod
    def new_basic_alpha_beta(cls, server_host, port, alpha, beta):
        assert alpha is not None
        assert beta is not None
        return cls( server_host, port,
                    alpha=alpha, beta=beta,
                    want_delta_updates=False)


    #@staticmethod
    #def splice_dropout_weights_to_L_param_desc(
    #    L_param_desc_with_maybe_dropout_probs,
    #    D_dropout_probs):
    #    # mutates the values in L_param_desc_with_dropout_probs
    #
    #    for param_desc in L_param_desc_with_maybe_dropout_probs:
    #        (layer_name, layer_number, role) = ClientCNNAutoSplitter.analyze_param_name(param_desc["name"])
    #        dropout_probs = D_dropout_probs[layer_name]
    #        param_desc['dropout_probs'] = dropout_probs

    @staticmethod
    def analyze_param_name(name):
        prog = re.compile(r"(layer_(\d+))_(.*)")
        m = prog.match(name)
        if m:
            return (m.group(1), int(m.group(2)), m.group(3))
        else:
            print "Failed to get layer number from %s." % name
            return None

    def perform_split(self, D_dropout_prob_pairs, L_extra_suffixes=[]):

        # D_dropout_prob_pairs is a dict with keys being layer names (e.g. "layer_0" and "layer_17").
        # The values are pairs of real numbers in [0.0,1.0]
        # indicating how much we want to drop out.
        #    zero means we keep everything.
        #    one means we drop all.

        # L_extra_suffixes is optional, and it contains things such as
        # "momentum" and "decay".
        # This means that we'll generate the appropriate splits for
        # the variables named
        #    ["layer_0_b_momentum", "layer_0_b_decay",
        #     "layer_0_W_momentum", "layer_0_W_decay"]

        if self.L_param_desc is None:
            self.read_param_desc_from_server()
            assert self.L_param_desc is not None

        self.splits_indices = sample_dropout_indices.sample_dropout_indices(self.L_params,
                                                                            D_dropout_prob_pairs,
                                                                            L_extra_suffixes)

    def pull_entire_param(self, name):

        param_desc = self.get_param_desc(name)
        assert param_desc is not None

        D = (param_desc['shape'][0], param_desc['shape'][1])
        dtype_for_client = DTYPE_FLOAT32 # from messages.py
        indices = (np.arange(0, D[0], dtype=np.intc), np.arange(0, D[1], dtype=np.intc))

        value = self.get_param_slice_from_server(name, D, D, indices, dtype_for_client)
        value = value.reshape(param_desc['shape'])

        return value

    # more for debugging purposes
    def push_entire_param(self, name, updated_value, alpha, beta):

        # maybe have an assertion here concerning the contiguous memory allocation requirement

        param_desc = self.get_param_desc(name)
        assert param_desc is not None

        D = (param_desc['shape'][0], param_desc['shape'][1])
        dtype_for_client = DTYPE_FLOAT32 # from messages.py
        indices = (np.arange(0, D[0], dtype=np.intc), np.arange(0, D[1], dtype=np.intc))

        updated_value = updated_value.astype(np.float32)

        self.update_param_slice_to_server(updated_value,
            alpha, beta,
            name,
            D, D, indices,
            dtype_for_client)


    def pull_split_param(self, name):

        indices = self.splits_indices[name]
        param_desc = self.get_param_desc(name)
        assert param_desc is not None


        S = (len(indices[0]), len(indices[1]))
        D = (param_desc['shape'][0], param_desc['shape'][1])
        dtype_for_client = DTYPE_FLOAT32 # from messages.py

        self.splits_timestamp_pull[name] = time.time()
        value = self.get_param_slice_from_server(name, S, D, indices, dtype_for_client)

        original_shape = (S[0], S[1], param_desc['shape'][2], param_desc['shape'][3])
        value = value.reshape(original_shape)

        # debug
        #if name == "layer_7_b":
        #    print "pull_split_param layer_7_b :"
        #    print value.reshape((-1,))


        # TODO :
        # if self.want_delta_updated  ... store the value in self.splits_cached_values[name]

        self.total_time_spent_pulling = self.total_time_spent_pulling + (time.time() - self.splits_timestamp_pull[name])
        return value

    
    def push_split_param(self, name, updated_value):
        # `updated_value` has the same shape as the `value` from `pull_split_param`.

        # maybe have an assertion here concerning the contiguous memory allocation requirement

        indices = self.splits_indices[name]
        param_desc = self.get_param_desc(name)
        assert param_desc is not None


        S = (len(indices[0]), len(indices[1]))
        D = (param_desc['shape'][0], param_desc['shape'][1])
        dtype_for_client = DTYPE_FLOAT32 # from messages.py

        updated_value = updated_value.astype(np.float32)

        self.splits_timestamp_push[name] = time.time()

        # Then we sent the updates to the server here.
        self.update_param_slice_to_server(updated_value,
            alpha, beta,
            name,
            S, D, indices,
            dtype_for_client)

        #self.total_time_spent_pushing = self.total_time_spent_pushing + (time.time() - self.splits_timestamp_push[name])

        # We're done. Nothing to return.
        return












