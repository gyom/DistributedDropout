

import struct
import numpy as np

MSG_TYPE_NULL = 0
MSG_TYPE_PULL_PARAM = 1
MSG_TYPE_PUSH_PARAM = 2
MSG_TYPE_CLIENT_QUITS = 3
MSG_TYPE_DISCONNECT = 4
MSG_TYPE_LIST_ALL_PARAMS_DESC = 5
MSG_TYPE_SAVE_ALL_TO_HDF5 = 6
MSG_TYPE_LOAD_ALL_FROM_HDF5 = 7


msg_type_dict = {'MSG_TYPE_NULL' : MSG_TYPE_NULL,
                 'MSG_TYPE_PULL_PARAM' : MSG_TYPE_PULL_PARAM,
                 'MSG_TYPE_PUSH_PARAM' : MSG_TYPE_PUSH_PARAM,
                 'MSG_TYPE_CLIENT_QUITS' : MSG_TYPE_CLIENT_QUITS,
                 'MSG_TYPE_DISCONNECT' : MSG_TYPE_DISCONNECT,
                 'MSG_TYPE_LIST_ALL_PARAMS_DESC' : MSG_TYPE_LIST_ALL_PARAMS_DESC,
                 'MSG_TYPE_SAVE_ALL_TO_HDF5':MSG_TYPE_SAVE_ALL_TO_HDF5,
                 'MSG_TYPE_LOAD_ALL_FROM_HDF5':MSG_TYPE_LOAD_ALL_FROM_HDF5
                 }

MSG_HEADER_LENGTH = 16
PARAM_NAME_LENGTH = 64
SLICE_MAX_INDEX = (256*256)

DTYPE_FLOAT16 = 16
DTYPE_FLOAT32 = 32
DTYPE_FLOAT64 = 64

dtype_int_to_numpy_dict = {DTYPE_FLOAT16 : np.float16,
                           DTYPE_FLOAT32 : np.float32,
                           DTYPE_FLOAT64 : np.float64}

dtype_int_to_size_dict  = {DTYPE_FLOAT16 : 2,
                           DTYPE_FLOAT32 : 4,
                           DTYPE_FLOAT64 : 8}



class MsgHeader(object):

    def __init__(self, msg_type_str):
        self.msg_type = msg_type_dict[msg_type_str]

    def encode(self):
        header = struct.pack("<B", np.uint8(self.msg_type))
        header = header + '\0' * (MSG_HEADER_LENGTH - len(header))
        assert len(header) == MSG_HEADER_LENGTH
        return header

    def send(self, conn):
        write_bytes(conn, self.encode())

class MsgSaveAllToHDF5(object):
    def __init__(self, pathHDF5):
        self.pathHDF5 = pathHDF5
        
    def encode(self):
        contents = ""
        contents += struct.pack("<i", np.int32(len(self.pathHDF5)))
        contents += self.pathHDF5

        return contents
    def send(self, conn):
        write_bytes(conn, self.encode())


class MsgLoadAllFromHDF5(object):
    def __init__(self, pathJSON, pathHDF5):
        self.pathJSON = pathJSON
        self.pathHDF5 = pathHDF5

    def encode(self):
        contents = ""
        contents += struct.pack("<i", np.int32(len(self.pathJSON)))
        contents += self.pathJSON
        contents += struct.pack("<i", np.int32(len(self.pathHDF5)))
        contents += self.pathHDF5

        return contents

    def send(self, conn):
        write_bytes(conn, self.encode())

class MsgPullParams(object):

    def __init__(self, name, N, D, indices, dtype_for_client):
        # name : string
        # indices : (I0, I1) where I0 and I1 are numpy arrays of integers
        # N : (N0, N1) integers
        # D : (D0, D1) integers
        
        self.name = name

        self.indices = [e if (e is not None) else np.zeros((0,), dtype=np.intc) for e in indices]
        assert len(self.indices) == 2
        assert self.indices[0].dtype == np.intc
        assert self.indices[1].dtype == np.intc

        self.N = N
        self.D = D
        assert len(N) == 2
        assert len(D) == 2
        
        self.dtype_for_client = dtype_for_client
        assert self.dtype_for_client in (DTYPE_FLOAT16, DTYPE_FLOAT32, DTYPE_FLOAT64)

    def encode(self):

        # add the null-termination for C-style strings
        contents = self.name + '\0'
        contents = contents + '\0' * (PARAM_NAME_LENGTH - len(contents))
        assert len(contents) == PARAM_NAME_LENGTH

        contents = contents + struct.pack("<iiiii",
            np.int32(self.dtype_for_client),
            np.int32(self.N[0]),
            np.int32(self.N[1]),
            np.int32(self.D[0]),
            np.int32(self.D[1]))

        # note that arrays of size 0 are just concatenated as '',
        # so this is compatible with the notation of having N[1] = 0
        # when self.indices[1] is empty
        contents = contents + self.indices[0].tostring() + self.indices[1].tostring()
        return contents

    def send(self, conn):
        write_bytes(conn, self.encode())

    def read_decode_response(self, conn):

        # read an int describing how many bytes will be sent next
        (response_nbr_bytes,) = struct.unpack("<i", read_bytes_as_string(conn, 4))
        assert 0 < response_nbr_bytes

        dtype = dtype_int_to_numpy_dict[self.dtype_for_client]
        elemsize = dtype_int_to_size_dict[self.dtype_for_client]
        count = response_nbr_bytes / elemsize
        
        numpy_array_decoded = np.fromstring(read_bytes_as_string(conn, response_nbr_bytes),
            dtype=dtype, count=count)
        return numpy_array_decoded



class MsgPushParams(MsgPullParams):

    def __init__(self, value, alpha, beta, name, N, D, indices, dtype_for_client):
        # value : numpy array (probably of np.float32)
        # alpha, beta : float
        # name : string
        # indices : (I0, I1) where I0 and I1 are numpy arrays of integers
        # N : (N0, N1) integers
        # D : (D0, D1) integers
        super(MsgPushParams, self).__init__(name, N, D, indices, dtype_for_client)
        self.value = value
        self.alpha = alpha
        self.beta = beta
        
    def encode(self):

        # This method looks more convoluted than it should, but it's because we
        # have to compute how many bytes are going to be sent.
        # The server will take that count at face value first,
        # but will later double-check to make sure that it's correct
        # given the slice used.

        # The conversion between a type and itself should be skipped by numpy (?).
        data_str = self.value.astype(dtype_int_to_numpy_dict[self.dtype_for_client]).tostring()

        ##
        ##    print "MsgPushParams wants to send %d bytes as data." % len(data_str)
        ##

        # same start as in the MsgPullParams
        contents = super(MsgPushParams, self).encode()
        # followed by the alpha, beta, current_data_length_bytes
        contents = contents + struct.pack("<ffi", self.alpha, self.beta, len(data_str))
        # and then the actual contents, converted to its declared type.
        contents = contents + data_str

        return contents

    def send(self, conn):
        write_bytes(conn, self.encode())

    def read_decode_response(self, conn):
        # not expecting a response from the server
        pass






class MsgListAllParamsDesc(object):

    def encode(self):
        return ''

    def send(self, conn):
        pass

    def read_decode_response(self, conn):
        (response_nbr_bytes,) = struct.unpack("<i", read_bytes_as_string(conn, 4))
        assert 0 < response_nbr_bytes

        ##
        ##    print "expecting %d byte to come from MsgListAllParamsDesc.read_decode_response" % response_nbr_bytes
        ##

        # this isn't the most exciting solution, but let's exchange data as json
        # because otherwise it'll be much more complicated and less flexible
        import json
        S = json.loads(read_bytes_as_string(conn, response_nbr_bytes))

        for s in S:
            # there is a slight hiccup in the msg encoding when we use unicode instead of ascii
            s['name'] = s['name'].encode('ascii')
            s['kind'] = s['kind'].encode('ascii')

        return S




def read_bytes_as_string(conn, nbr_bytes):

    # THINK : You might want to use another constant than 2048 because
    # then you end up partitionning large numpy arrays into chunks of 2048 bytes.
    # Maybe worth investigating.

    # boilerplate code from https://docs.python.org/2/howto/sockets.html
    chunks = []
    bytes_recd = 0
    while bytes_recd < nbr_bytes:
        chunk = conn.recv(min(nbr_bytes - bytes_recd, 2048))
        if chunk == '':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd = bytes_recd + len(chunk)

    return ''.join(chunks)

def write_bytes(conn, contents, nbr_bytes=None):

    if nbr_bytes is None:
        nbr_bytes = len(contents)

    # boilerplate code from https://docs.python.org/2/howto/sockets.html
    totalsent = 0
    while totalsent < nbr_bytes:
        sent = conn.send(contents[totalsent:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        totalsent = totalsent + sent






