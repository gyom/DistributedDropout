
#ifndef __COMMON_H__
#define __COMMON_H__

/* define _BSD_SOCKLEN_T_ in order to define socklen_t on darwin */
# define _BSD_SOCKLEN_T_
#include <sys/socket.h>
/* one more header to be included on OSX */
#include <netinet/in.h>
#include <stdarg.h>
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>
#include <glib.h>

#define MSG_HEADER_LENGTH 16
#define PARAM_NAME_LENGTH 64
#define SLICE_MAX_INDEX (256*256)

#define DTYPE_FLOAT16 16
#define DTYPE_FLOAT32 32
#define DTYPE_FLOAT64 64

#define FULLY_CONNECTED_WEIGHTS 1
#define FULLY_CONNECTED_BIASES 2
#define CONV_FILTER_WEIGHTS 3
#define CONV_FILTER_BIASES 4

struct _param_t;

typedef struct _param_t {
	char name[PARAM_NAME_LENGTH];
	int shape[4];
	int kind; // conv weights, bias, full weights, bias
	pthread_mutex_t mutex;
	void * data;
	int data_length_bytes; // redundant but nice to have
	int dtype;
	struct _param_t * next;
} param_t;

// The slices are not small. They are 128K in size.
// This would be even worse if we couldn't represent
// them as cartesian product.
typedef struct _slice_t {
	int indices[2][SLICE_MAX_INDEX];
	int S[2]; // the number of indices represented in the slice
	int D[2]; // the total number of indices in the original array
} slice_t;


typedef struct _msg_header_t {
	int msg_type;

	/* a space to read the header before parsing it */
	char buffer[MSG_HEADER_LENGTH];
} msg_header_t;

typedef struct _msg_param_t {
	char name[PARAM_NAME_LENGTH];
	int dtype_for_client;
	slice_t slice;
	float alpha, beta;
	void * data;
	int current_data_length_bytes;
	int max_data_length_bytes;
	// int msg_type; //optional, to track what's stored in there
} msg_param_t;


#define MSG_TYPE_NULL 0
#define MSG_TYPE_PULL_PARAM 1
#define MSG_TYPE_PUSH_PARAM 2
#define MSG_TYPE_CLIENT_QUITS 3
#define MSG_TYPE_DISCONNECT 4
#define MSG_TYPE_LIST_ALL_PARAMS_DESC 5
#define MSG_TYPE_SAVE_ALL_TO_HDF5 6
#define MSG_TYPE_LOAD_ALL_FROM_HDF5 7

typedef struct _thread_liveness_t{
    pthread_t thread_id;
    sig_atomic_t is_alive;
} thread_liveness_t;


typedef struct _client_conn_t {
    /* the socket connection part */
    socklen_t address_length;
    int socket_fd;
    struct sockaddr_in client;

    /* the threading part */
    pthread_t thread_id;
    volatile bool want_thread_to_stop;
	sig_atomic_t * is_alive;
    /* instead of using a global variable for
       the shared linked list of global parameters,
       we'll include it in here even though it doesn't
       belong with a particular client connection */
	param_t * global_param_list;

	int max_data_length_bytes;

} client_conn_t;

void thread_printf(char * format, ...);
void log_info(size_t fd, char * str, size_t len);

#endif


