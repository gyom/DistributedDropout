

import numpy as np

import distdrop
import distdrop.client

from distdrop.client.client_api import Client
from distdrop.client import messages


def test_slice(client):

    shape = (1, 32, 2, 2)

    D = (4,6)
    S = (4,6)

    indices0 = np.arange(0, D[0], dtype=np.intc)
    np.random.shuffle(indices0)
    indices0 = indices0[0:S[0]]

    indices1 = np.arange(0, D[1], dtype=np.intc)
    np.random.shuffle(indices1)
    indices1 = indices1[0:S[1]]

    indices = (indices0, indices1)


    A = client.get_param_slice_from_server("debug_weights",
            S, D,
            indices,
            messages.DTYPE_FLOAT32)

    B = np.random.rand(*A.shape)
    #B = np.ones(A.shape)
    #(alpha, beta) = (0.9, 0.2)
    (alpha, beta) = (0.9, 1.2)


    client.update_param_slice_to_server(
        B,
        alpha, beta,
        "debug_weights",
        S, D, indices,
        messages.DTYPE_FLOAT32)

    C = client.get_param_slice_from_server("debug_weights",
            S, D,
            indices,
            messages.DTYPE_FLOAT32)

    Cref = alpha*A + beta*B
    diff = np.max(np.abs(C-Cref))
    print "Maximum abs difference is %f." % diff


def run():
    server_host = "127.0.0.1"
    port = 5005

    client = Client(server_host, port)

    client.connect()
    print client.read_param_desc_from_server()


    test_slice(client)

    
    client.quit()
    client.close()


if __name__ == "__main__":
    run()







