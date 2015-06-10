

#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include <unistd.h>

#include "common.h"
#include "handler.h"
#include "params.h"
#include "server_hdf5_io.h"

#include <jansson.h>
#include <glib.h>
#include <assert.h>



//////////////////////////////////////////////////////////////////////
// cleanup :
//   Free the buffers we malloc'd and try to close the socket if it's
//   still open. Set the thread liveness flag to false.
//////////////////////////////////////////////////////////////////////
void cleanup(msg_param_t * msg, msg_header_t * header, client_conn_t * conn) {
		printf("handler.c - pthread #%lu: Reached the cleanup tag in server_side_handler.\n", (size_t)pthread_self());

		// http://www.gnu.org/software/libc/manual/html_node/Closing-a-Socket.html
		if (shutdown(conn->socket_fd, 2) == -1) {
			const char * errorType = NULL;
			switch(errno) {
				case EBADF:
					errorType = "EBADF : socket is not a valid file descriptor.";
					break;

				case ENOTSOCK:
					errorType = "ENOTSOCK : socket is not a socket.";
					break;
				
				case ENOTCONN:
					errorType = "ENOTCONN : socket is not connected.";
					break;

				default:
					errorType = "Other";
					break;
			}
			// ENOTCONN is not really an error, it only means that the client has already closed
			// the socket
			if (errno != ENOTCONN) {
				printf("handler.c - pthread #%lu: shutdown returned errno #%d:\n\terror message: %s\n", (size_t)pthread_self(), errno, errorType);
			}
		}

    	if (msg->data) {
    		free(msg->data);
    	}
		if (msg) {
			free(msg);
		}
		if (header) {
			free(header);
		}
		
		printf("handler.c - pthread #%lu: setting is_alive to false\n", (size_t)pthread_self());
		*conn->is_alive = false;

		printf("handler.c - pthread #%lu: Finished the cleanup segment in server_side_handler.\n", (size_t)pthread_self());
}

//////////////////////////////////////////////////////////////////////
// fail:
//   We failed to get what we wanted, cleanup and early exit follow.
//////////////////////////////////////////////////////////////////////
void fail(const char * message, int len) {
	// Format the error message.
	// Adds tabs after line returns that aren't already followed
	// by tabs.
	// Might look like slight overkill, but I expect the load of
	// the server to be very low, and having clean looking error messages
	// can be quite a plus to the development effort.
	GError *errNew = NULL;
	GError *errReplace = NULL;
	char * messageCleaned = NULL;
	
	// Limit the length of the string to prevent
	// whole server crashing segfault 
	char * limitedMessage = g_strndup(message, len);

	GRegex * pattern = g_regex_new("\n(?!\t)", 0, 0, &errNew);
	if (!errNew && pattern) {
		messageCleaned = g_regex_replace(
			pattern, 
			limitedMessage, 
			-1, 
			0, 
			"\n\t\t", 
			0, 
			&errReplace);
	}

	char * generic_message = g_strdup_printf("handler.c - pthread #%lu: server_side_handler failed.\n\tError message:\n", (size_t)pthread_self());

	// If the cleaning up succeeded, print the cleaned up string
	if (!errNew && !errReplace) {
	fprintf(stderr, 
		"%s"
		"\t%s\n"
		, generic_message, messageCleaned);
	} 
	// If not, print the string received as an argument
	else {
	fprintf(stderr, 
		"%s"
		"\t%s\n"
		, generic_message, limitedMessage);
	}

	// .. g_free is a version of free that checks first if the pointer is NULL, 
	// and just returns without freeing if so.
	g_free(generic_message);
	g_free(limitedMessage);
	g_free(errNew);
	g_free(errReplace);
	g_free(messageCleaned);
	g_free(pattern);
}

void * server_side_handler(void * void_conn_ptr) {
	
	printf("handler.c - pthread #%lu: entering server_side_handler\n", (size_t)pthread_self());
    client_conn_t * conn = void_conn_ptr; 
	param_t * global_param_list = conn->global_param_list;

	// Try to allocate memory for the header and the message struct.
	// We will recycle the same memory at every pass through the loop
	msg_header_t * header = (msg_header_t *)malloc(sizeof(msg_header_t));
	msg_param_t * msg = (msg_param_t *)malloc(sizeof(msg_param_t));
	if (header == NULL || msg == NULL) {
		// Might seem silly to pass the len if we are just calling strlen,
		// but we know we can because we are allocating it ourselves.
		// This is not always true, and fail() has no way to know that for sure.
		const char * error_text = "Allocating memory for the message's header or\nparam struct failed. Shutting down the connection.";
		fail(error_text, strlen(error_text));
		cleanup(msg, header, conn);
		return NULL;
	}
	// Try to allocate memory for the parameter data
	msg->data = malloc(conn->max_data_length_bytes);
	if (msg->data == NULL) {
		// Might seem silly to pass the len if we are just calling strlen,
		// but we know we can because we are allocating it ourselves.
		// This is not always true, and fail() has no way to know that for sure.
		const char * error_text = "Allocating memory for the message's parameter\ndata failed. Shutting down the connection.";
		fail(error_text, strlen(error_text));
		cleanup(msg, header, conn);
		return NULL;
	}

	msg->max_data_length_bytes = conn->max_data_length_bytes;

	// matched_param is never allocated, never deallocated, always a pointing to something
	// that is found through exploring the `global_param_list`
	param_t * matched_param = NULL;

	int total_time_waited_until_header = 0;
	int short_sleep_time = 25*1000;
	int long_sleep_time = 500*1000;

	// when we don't know if we have data that should be waiting, and the connection might be dead,
   	// but we're willing to wait 20 minutes before declaring it as dead 
	int failure_impromptu_data_timeout = 20*60*1000*1000;
	// when we know that we have data that should be waiting, but it's not coming for 30 seconds 
	int failure_promised_data_timeout = 30*1000*1000;

	// Server-client interaction loop
	while (1) {
		//////////////////////////////////////////////////////////////////////
		// Verify if the client wants to close the connection
		//////////////////////////////////////////////////////////////////////
		if (conn->want_thread_to_stop) {
			printf("handler.c - pthread #%lu: exiting server_side_handler from setting conn->want_thread_to_stop externally\n", (size_t)pthread_self());
			cleanup(msg, header, conn);
			return NULL;
		}

		/* As a result of reading the `header`, this connection might now
	       be closed. If we can't read any more bytes, then the header will
	       be put into MSG_TYPE_DISCONNECT.
		*/

		/*  The reasoning is the following :
			We are willing to wait a long time for a header to arrive
			because we don't know when a new header is coming.
			We are NOT willing to wait a long time when a payload
			is supposed to arrive but it's not arriving.
			In that case, we have to be very pro-active.
		 */

		//////////////////////////////////////////////////////////////////////
		// Verify if we have some data to interpret
		//////////////////////////////////////////////////////////////////////
	    int bytes_available = recv(conn->socket_fd, header->buffer, MSG_HEADER_LENGTH, MSG_PEEK);	    
	    if (bytes_available < 0) {
	    	const char * error_text = "Got a negative number with MSG_PEEK";
	    	fail(error_text, strlen(error_text));
	    	cleanup(msg, header, conn);
			return NULL;
	    } else if (bytes_available == 0) {
	    	/* The first time around when we get to 0 bytes,
	    	   we want to sleep just for a quick nap.
	    	   This doesn't make as much sense as I hoped it would. 
	    	*/
	    	if (total_time_waited_until_header == 0) {
	    		usleep(short_sleep_time);
	    		total_time_waited_until_header += short_sleep_time;
	    	} else {
	    		usleep(long_sleep_time);
	    		total_time_waited_until_header += long_sleep_time;
	    	}

	    	if (failure_impromptu_data_timeout < total_time_waited_until_header) {
	    		char * error_text = g_strdup_printf("Main loop waiting much too long for a header.\nGot nothing for %d minutes so we're calling this connection dead", total_time_waited_until_header / (60*1000*1000));
	    		// g_strdup_printf is a safe function we can call strlen on.
	    		fail(error_text, strlen(error_text));
	    		free(error_text);
	    		cleanup(msg, header, conn);
				return NULL;
	    	}
	    	continue;
	    	
	    } else if (0 < bytes_available) {

	    	/* yay ! we have something to read */
	    	total_time_waited_until_header = 0;

	    	// proceed to the main body of the header reading and parsing

	    }
		
	    //////////////////////////////////////////////////////////////////////
		// We confirmed that we have some data to interpret.
		// We start doing so.
		//////////////////////////////////////////////////////////////////////

		// We reset the previous values we had read, or we initialize them for the first time
		clean_msg_header(header);
		clean_msg_param(msg);

		// We receive and extract the header number
	    int header_bytes_read = read_MSG_HEADER(header, conn->socket_fd);
	    if (header_bytes_read != MSG_HEADER_LENGTH) {
	    	char * error_text = g_strdup_printf("We tried to read a message header, but we got %d bytes instead of %d.\n", header_bytes_read, MSG_HEADER_LENGTH);
	    	// g_strdup_printf is a safe function we can call strlen on
	    	fail(error_text, strlen(error_text));
	    	free(error_text);
	    	cleanup(msg, header, conn);
			return NULL;
	    }

		/* DEVEL : let's be cautious about whether or not we have to reset
                   the contents of the header at every iteration
		*/

        // We act according to the header number
        switch(header->msg_type){
        	// The client wants a set of parameters to train with
			case MSG_TYPE_PULL_PARAM: 
				printf("handler.c - pthread #%lu: read header for MSG_TYPE_PULL_PARAM\n", (size_t)pthread_self());
				// extract the body of the message
				read_MSG_PULL_PARAM(msg, conn->socket_fd);
				// fetch the param by name. 
				matched_param = get_matching_param_entry(global_param_list, msg->name);
				if (matched_param == NULL) {
					// Give the client something to realize that it has made a mistake.
					// Send in an empty array, of length zero.
					int zero = 0;
					write(conn->socket_fd, (void *)&zero, sizeof(int));

					// We warn that we did not find any matching params
					char * error_text = g_strdup_printf(
									"Error for MSG_TYPE_PULL_PARAM.\n"
									"You asked for parameter %s but there is no such parameter on the server.\n"
									"It seems a bit excessive to terminate the connection for just that recoverable error,\n"
									"but it really means that you have a bug in your client code because you expected \n"
									"that parameter to be present.\n"
									"Therefore, we terminate the connection on the server side."
									, msg->name);

					fail(error_text, strlen(error_text));
					free(error_text);
	    			cleanup(msg, header, conn);
					return NULL;
				}
				if(extract_slice_from_param(matched_param, msg) == -1) {
					// Invalid slice
					const char * error_text = "Extracting a slice from the parameter struct failed.";
					fail(error_text, strlen(error_text));
	    			cleanup(msg, header, conn);
					return NULL;
				}
				respond_MSG_PULL_PARAM(msg, conn->socket_fd);
				
				printf("handler.c - pthread #%lu: done with MSG_TYPE_PULL_PARAM;\n", (size_t)pthread_self());			
			break;
			// The client has updated params to push to the server.
			case MSG_TYPE_PUSH_PARAM:
				printf("handler.c - pthread #%lu: read header for MSG_TYPE_PUSH_PARAM\n", (size_t)pthread_self());
				// Read the body of the message
				read_MSG_PUSH_PARAM(msg, conn->socket_fd);
				// Find the param by name
				matched_param = get_matching_param_entry(global_param_list, msg->name);
				if (matched_param == NULL) {
					// Give the client something to realize that it has made a mistake.
					// Send in an empty array, of length zero.
					int zero = 0;
					
					write(conn->socket_fd, (void *)&zero, sizeof(int));				
					char * error_text = g_strdup_printf(
											"Error for MSG_TYPE_PUSH_PARAM.\n"				
											"You asked for parameter %s but there is no such parameter on the server.\n"
											"It seems a bit excessive to terminate the connection for just that recoverable error,\n"
											"but it really means that you have a bug in your client code because you expected that\n"
											"parameter to be present.\n"
											"Therefore, we terminate the connection on the server side."
											, msg->name);

					fail(error_text, strlen(error_text));
					free(error_text);
	    			cleanup(msg, header, conn);
					return NULL;
				}

				// Update the server's corresponding param slice 
				commit_slice_to_param(matched_param, msg);

				// no need to respond here (this can be changed later if we tweak the protocol)
				
				printf("handler.c - pthread #%lu: done with MSG_TYPE_PUSH_PARAM;\n", (size_t)pthread_self());
			break;
			// The client wants all of the parameters
			case MSG_TYPE_LIST_ALL_PARAMS_DESC:
				printf("handler.c - pthread #%lu: read header for MSG_TYPE_LIST_ALL_PARAMS_DESC\n", (size_t)pthread_self());

				// TODO : implement a proper response
				respond_MSG_LIST_ALL_PARAMS_DESC(global_param_list, conn->socket_fd);
			break;
			// The client wants to close the connection.
			case MSG_TYPE_CLIENT_QUITS:
				printf("handler.c - pthread #%lu: read header for MSG_TYPE_CLIENT_QUITS\n", (size_t)pthread_self());
					    		
	    		cleanup(msg, header, conn);
				return NULL;
			break; 
			// The client wants to disconnect.
			case MSG_TYPE_DISCONNECT:
				printf("handler.c - pthread #%lu: read header for MSG_TYPE_DISCONNECT\n", (size_t)pthread_self());

	    		cleanup(msg, header, conn);
				return NULL;
			break;
			case MSG_TYPE_SAVE_ALL_TO_HDF5:
				printf("handler.c - pthread #%lu: read header for MSG_TYPE_SAVE_ALL_TO_HDF5\n", (size_t)pthread_self());
				read_MSG_TYPE_SAVE_ALL_TO_HDF5(global_param_list, conn->socket_fd);
			break;

			case MSG_TYPE_LOAD_ALL_FROM_HDF5:
				printf("handler.c - pthread #%lu: read header for MSG_TYPE_LOAD_ALL_FROM_HDF5\n", (size_t)pthread_self());
				read_MSG_TYPE_LOAD_ALL_FROM_HDF5(&global_param_list, conn->socket_fd);
			break;

			// Unknown message.
			default:
				// C doesn't allow declaration of variable in a switch case without surrounding it's use by brackets to mark its scope.
				// The compiler errors for that are really unexpressive. They're actually better in C++, for that specific case.
				{
				char * error_text = g_strdup_printf("Received unknown message header: '%d'\nQuitting.", header->msg_type);
		    	fail(error_text, strlen(error_text));
		    	free(error_text);
				cleanup(msg, header, conn);
				return NULL;
				}
			break;
		}
	}

	cleanup(msg, header, conn);
	return NULL;
}

int validate_slice(slice_t * slice_ptr, int shape[4], char * name) {

	/* little check to catch silly mistakes */

	for (int k = 0; k < 2; k++) {
		if (shape[k] != slice_ptr->D[k]) {
			fprintf(stderr, "handler.c - pthread #%lu: Parameter %s has shape[%d]=%d, but you expected %d.\n",
				(size_t)pthread_self(),
				name,
				k,
				shape[k],
				slice_ptr->D[k]);
			return -1;
		}

		if (slice_ptr->D[k] < slice_ptr->S[k]) {
			fprintf(stderr, "handler.c - pthread #%lu: Query for parameter %s involves bad values of (S,D).\n", (size_t)pthread_self(), name);
			fprintf(stderr, "handler.c - pthread #%lu: We should not have that D[%d]:%d  <  S[%d]:%d\n", (size_t)pthread_self(), k, slice_ptr->D[k], k, slice_ptr->S[k]);
			return -1;
		}

        /*
          We can perform the more expensive check to insure that the indices are ordered
          and within the bounds that we expect. We don't really need them to be ordered,
          but they will be ordered by construction, so we might as well test that as sanity check.
          This feels a bit wasteful, but it's really not that expensive compared to the cost
          of using the cartesian product of the indices.
        */

        for (int i = 0; i < slice_ptr->S[k]; i++) {

            // test if we're within the bounds
            if (0 <= slice_ptr->indices[k][i] && slice_ptr->indices[k][i] < slice_ptr->D[k]) {

                // test if we have a strictly increasing ordered list of indices
                if (0 < i) {
                    if (slice_ptr->indices[k][i-1] < slice_ptr->indices[k][i]) {
                        // all ok
                        continue;
                    } else {
                        fprintf(stderr, "handler.c - pthread #%lu: Got a slice for param %s that has un-ordered indices.\n", (size_t)pthread_self(), name);
                        return -1;
                    }
                }
            } else {
              fprintf(stderr, "handler.c - pthread #%lu: Got a slice for param %s that has indices that are out of bounds ! indices[%d][%d] is %d but should be limited to %d.\n", (size_t)pthread_self(), name, k, i, slice_ptr->indices[k][i], slice_ptr->D[k]);
                return -1;
            }

        }
	}

	return 0;
}

int extract_slice_from_param(param_t * matched_param, msg_param_t * msg) {

	if (validate_slice(&msg->slice, matched_param->shape, matched_param->name) == -1) {
		return -1;
	}

	// for some reason the compiler refuses to take those
	// declarations inside the switch
	float * dst = (float *)msg->data;
	float * src = (float *)matched_param->data;

	int nbr_subelements = matched_param->shape[2]*matched_param->shape[3];

	switch (msg->dtype_for_client) {

		case DTYPE_FLOAT16:
			fprintf(stderr, "handler.c - pthread #%lu: Error. The FLOAT16 case for dtype_for_client is not implemented yet.\n", (size_t)pthread_self());
			return -1;
			
		case DTYPE_FLOAT64:
			fprintf(stderr, "handler.c - pthread #%lu: Error. The FLOAT64 case for dtype_for_client is not implemented (and probably will never be).\n", (size_t)pthread_self());
			return -1;

		case DTYPE_FLOAT32:

			msg->current_data_length_bytes = msg->slice.S[0] * msg->slice.S[1] * nbr_subelements * sizeof(float);
			if (msg->max_data_length_bytes < msg->current_data_length_bytes) {
			 	printf(	"handler.c - pthread #%lu: Error. The MSG_PULL_PARAM announced that it wants to retrieve %d bytes,"
			 			"but with the current buffers set up, we can at most deal with %d bytes.\n",
			 			(size_t)pthread_self(),
			 			msg->current_data_length_bytes,
			 			msg->max_data_length_bytes);
			 	/* Note that this isn't insurmountable. We could just have smaller buffers,
			        use less memory, and we would be fine. Scaling to 100+ clients we might
			        be better with a different approach.
			 	*/
			 	return -1;
			}

			if ((msg->slice.D[0] == msg->slice.S[0]) && (msg->slice.D[1] == msg->slice.S[1])) {

				/* if we're not REALLY doing slices, then we might as well use memcpy */
				memcpy(dst, src, msg->slice.D[0] * msg->slice.D[1] * nbr_subelements * sizeof(float));
			} else {

				// TODO : add error-checking here
				extract_slice_to_param_float32_to_float32(
					matched_param->mutex,
					&msg->slice,
					src,
					dst,
					nbr_subelements);
			}
			break;
	}

	return 0;
}
			
int commit_slice_to_param(param_t * matched_param, msg_param_t * msg) {

	if (validate_slice(&msg->slice, matched_param->shape, matched_param->name) == -1) {
		return -1;
	}

	//printf("At the start of commit_slice_to_param, we have (alpha=%f, beta=%f).\n", msg->alpha, msg->beta);

	// for some reason the compiler refuses to take those
	// declarations inside the switch
	float * dst = (float *)matched_param->data;
	float * src = (float *)msg->data;

	int nbr_subelements = matched_param->shape[2]*matched_param->shape[3];

	switch (msg->dtype_for_client) {

		case DTYPE_FLOAT16:
			fprintf(stderr, "handler.c - pthread #%lu: Error. The FLOAT16 case for dtype_for_client is not implemented yet.\n", (size_t)pthread_self());
			return -1;
			
		case DTYPE_FLOAT64:
			fprintf(stderr, "handler.c - pthread #%lu: Error. The FLOAT64 case for dtype_for_client is not implemented (and probably will never be).\n", (size_t)pthread_self());
			return -1;

		case DTYPE_FLOAT32:

			/*  We trusted the client converning the value of `current_data_length_bytes`,
				but now that we have matched with the parameter, we can double-check to make
				sure that it's correct.
			*/
			if ( msg->current_data_length_bytes != msg->slice.S[0] * msg->slice.S[1] * nbr_subelements * sizeof(float)) {
				fprintf(stderr, "handler.c - pthread #%lu: Error in commit_slice_to_param.\n"
								"We were told by the client that the slice of parameter %s would take %d bytes.\n"
								"Instead of that, the server-side calculates that it should take %zu bytes.\n",
								(size_t)pthread_self(),
								matched_param->name, msg->current_data_length_bytes,
								msg->slice.S[0] * msg->slice.S[1] * nbr_subelements * sizeof(float)	);
				return -1;
			}


			if ((msg->slice.D[0] == msg->slice.S[0]) && (msg->slice.D[1] == msg->slice.S[1]) && (msg->alpha==1.0) && (msg->beta==0.0)) {

				/* if we're not REALLY doing slices, and we have (alpha=1.0, beta=0.0),
				   then we might as well use memcpy */
				memcpy(dst, src, msg->slice.D[0] * msg->slice.D[1] * nbr_subelements * sizeof(float));
			} else {

				// TODO : add error-checking here
				commit_slice_to_param_float32_to_float32(
					matched_param->mutex,
					&msg->slice,
					src,
					dst,
					nbr_subelements,
					msg->alpha, msg->beta );

			}
			break;
	}



	return 0;
}

param_t * get_matching_param_entry(param_t * global_param_list, char * name) {

	param_t * current = global_param_list;
	while (current != NULL) {
		if (strcmp(current->name, name) == 0) {
			// found it
			//printf("DEBUG : found parameter %s\n", name);
			return current;
		} else {
			//printf("DEBUG : %s doesn't match %s. going to next parameter.\n", current->name, name);
			current = current->next;
		}
	}
	// return NULL to indicate failure to find
	return NULL;
}

void clean_msg_header(msg_header_t * header) {
	header->msg_type = MSG_TYPE_NULL;
}

void clean_msg_param(msg_param_t * msg) {
	msg->name[0] = '\0';
	msg->dtype_for_client = 0;
	msg->alpha = 1.0;
	msg->beta = 0.0;
	msg->current_data_length_bytes = 0;
}



int block_on_recv(int socket_fd, void * buffer, int len) {
	/* This function was born out of frustration with calls to
	   read/recv that just would return "Operation timed out"
	   and report 0 bytes being read. Somehow we can't manage
	   to set a socket to be blocking.
	*/

	int bytes_read = 0;
	int bytes_left = len;
	size_t noOfReads = 0;
	int total_bytes_read = 0;

	while (0 < bytes_left) {
		bytes_read = recv(socket_fd, buffer, bytes_left, MSG_WAITALL);
		assert(bytes_read >= 0);
		buffer += bytes_read;
		bytes_left -= bytes_read;
		total_bytes_read += bytes_read;
	}
	
	if (len != total_bytes_read) {
		printf(">>>>>>>>>>>>>>>>> Blocking read didn't read the right amount");
	}

	return total_bytes_read;
}



int read_MSG_HEADER(msg_header_t * header, int socket_fd) {
	block_on_recv(socket_fd, (void *)header->buffer, MSG_HEADER_LENGTH);
	header->msg_type = ((int *)header->buffer)[0];
	return MSG_HEADER_LENGTH;
}


int read_MSG_PULL_PARAM(msg_param_t * msg, int socket_fd) {

	if (block_on_recv(socket_fd, (void *)msg->name, PARAM_NAME_LENGTH) != PARAM_NAME_LENGTH) { return -1; }
	if (block_on_recv(socket_fd, (void *)&msg->dtype_for_client, sizeof(int)) != sizeof(int)) { return -1; }
	if (block_on_recv(socket_fd, (void *)msg->slice.S, 2*sizeof(int)) != 2*sizeof(int)) { return -1; }
	if (block_on_recv(socket_fd, (void *)msg->slice.D, 2*sizeof(int)) != 2*sizeof(int)) { return -1; }


	int elemsize = 0;
	switch (msg->dtype_for_client) {
	case DTYPE_FLOAT16:
		elemsize = sizeof(float) / 2;
		break;
	case DTYPE_FLOAT32:
		elemsize = sizeof(float);
		break;
	case DTYPE_FLOAT64:
		elemsize = sizeof(double);
		break;
	default:
		printf("handler.c - pthread #%lu: Error. Illegal msg->dtype_for_client : %d.\n", (size_t)pthread_self(), msg->dtype_for_client);
		return -1;
	}

	/* 	This isn't the right location to infer the `current_data_length_bytes`.
		We are missing the contributions of shape[3] * shape[4], which we don't
		have because we haven't searched for the parameter.
	 */
	// msg->current_data_length_bytes = msg->slice.S[0] * msg->slice.S[1] * elemsize;
	// if (msg->max_data_length_bytes < msg->current_data_length_bytes) {
	// 	printf(	"Error. The MSG_PULL_PARAM announced that it wants to retrieve %d bytes,"
	// 			"but with the current buffers set up, we can at most deal with %d bytes.\n",
	// 			msg->current_data_length_bytes,
	// 			msg->max_data_length_bytes);
	// 	/* Note that this isn't insurmountable. We could just have smaller buffers,
	//        use less memory, and we would be fine. Scaling to 100+ clients we might
	//        be better with a different approach.
	// 	*/
	// 	return -1;
	// }

	/* the goal here isn't to populate the whole msg->slice.indices[0][:],
	   but only to read as many elements as we have elements waiting
	*/
	if (block_on_recv(socket_fd, (void *)msg->slice.indices[0], sizeof(int) * msg->slice.S[0]) != sizeof(int) * msg->slice.S[0]) { return -1; }
	if (block_on_recv(socket_fd, (void *)msg->slice.indices[1], sizeof(int) * msg->slice.S[1]) != sizeof(int) * msg->slice.S[1]) { return -1; }

	return 0;
}

int read_MSG_PUSH_PARAM(msg_param_t * msg, int socket_fd) {

	// Both methods start the same, so we might as well reuse the code.
	// Be careful about adding more stuff to read_MSG_PULL_PARAM, now.
	read_MSG_PULL_PARAM(msg, socket_fd);

	if (block_on_recv(socket_fd, (void *)&(msg->alpha), sizeof(float)) != sizeof(float)) { return -1; }
	if (block_on_recv(socket_fd, (void *)&(msg->beta),  sizeof(float)) != sizeof(float)) { return -1; }
	if (block_on_recv(socket_fd, (void *)&(msg->current_data_length_bytes), sizeof(int)) != sizeof(int)) { return -1; }

	//printf("read_MSG_PUSH_PARAM sees (alpha=%f, beta=%f).\n", msg->alpha, msg->beta);

	if (msg->max_data_length_bytes < msg->current_data_length_bytes) {
	 	printf(	"handler.c - pthread #%lu: Error. The MSG_TYPE_PUSH_PARAM announced that it wants to send %d bytes,"
	 			"but with the current buffers set up, we can at most deal with %d bytes.\n",
	 			(size_t)pthread_self(),
	 			msg->current_data_length_bytes,
	 			msg->max_data_length_bytes);
	 	/* Note that this isn't insurmountable. We could just have smaller buffers,
	        use less memory, and we would be fine. Scaling to 100+ clients we might
	        be better with a different approach.
	 	*/
		return -1;
	}

	/* read the actual data transferred */
	printf("handler.c - pthread #%lu: read_MSG_PUSH_PARAM expecting %d bytes as data\n", (size_t)pthread_self(), msg->current_data_length_bytes);
	if (block_on_recv(socket_fd, (void *)msg->data, msg->current_data_length_bytes) != msg->current_data_length_bytes) { return -1; }

	return 0;
}

int respond_MSG_PULL_PARAM(msg_param_t * msg, int socket_fd) {

	/*  We would keep the header the same and resend it,
        but it's not necessary at all.

        Don't send the slice info, don't send anything else.
        Just send the data right away. The sender knows what
        to expect because the sender should be holding to that
        information.

        This can be modified easily.
	*/

    if (write(socket_fd, (void *)&msg->current_data_length_bytes, sizeof(int)) != sizeof(int)) { return -1; }
    if (write(socket_fd, (void *)msg->data, msg->current_data_length_bytes) != msg->current_data_length_bytes) { return -1; }

	return 0;
}

int respond_MSG_LIST_ALL_PARAMS_DESC(param_t * global_param_list, int socket_fd) {

	char * response = NULL;
	char * bad_response = "[]";
	json_t * enc = encode_list_param_to_json_t(global_param_list);
	if ( enc == NULL ) {
		fprintf(stderr, "handler.c - pthread #%lu: Failed to get the global param list as json_t object in respond_MSG_LIST_ALL_PARAMS_DESC.\n", (size_t)pthread_self());
		response = bad_response;
	} else {
		response = json_dumps(enc, JSON_PRESERVE_ORDER);		
	}

	int response_length = strlen(response);
	if (write(socket_fd, (void *)&response_length, sizeof(int)) != sizeof(int)) { return -1; }
	if (write(socket_fd, (void *)response, response_length) != response_length) { return -1; }

	if (response != bad_response) { free(response); }

	/* cleanup enc properly in jansson (I think that's how it's done) */
	json_decref(enc);

	return 0;
}



int read_MSG_TYPE_SAVE_ALL_TO_HDF5(param_t * global_param_list, int socket_fd) {
	// Receive the length of the filename
	const GRegex * path_cleaning_pattern = g_regex_new("[^\\w]", 0, 0, 0);
	int path_length = 0;
	if (block_on_recv(socket_fd, (void *) &path_length, sizeof(int)) != sizeof(int)) 
		{
			return -1;
		}
	// Receive the filename
	char * received_path = malloc(sizeof(char) * path_length);
	if (block_on_recv(socket_fd, (void *) received_path, path_length) != path_length) 
		{
			return -1;
		}
	
	// Sanitize the filename by purging anything but alphanumeric characters (regex \w)
	char * cleaned_path = g_regex_replace(path_cleaning_pattern, received_path, strlen(received_path), 0, "", 0, NULL);
	// This part is not necessary, but makes for easier code to understand 
	char * cwd = malloc(100 * sizeof(char));
	getcwd(cwd, 99);
	cwd[99] = 0; 
	// end of unnecess. code
	char * final_path = g_strdup_printf ("%s/%s.hdf5", cwd, cleaned_path);
	printf("Final path: '%s'\n", final_path);

	locking_save_to_hdf5(global_param_list, final_path);

	g_free(final_path);
	g_free(cleaned_path);
	g_free(received_path);

	return 0;
}


int read_MSG_TYPE_LOAD_ALL_FROM_HDF5(param_t ** global_param_list, int socket_fd) {
	const GRegex * path_cleaning_pattern = g_regex_new("[^\\w]", 0, 0, 0);

	// Receive the length of the filename
	int json_path_length = 0;
	if (block_on_recv(socket_fd, (void *) &json_path_length, sizeof(int)) != sizeof(int)) 
		{
			return -1;
		}
		
	// Receive the filename
	char * received_json_path = malloc(sizeof(char) * json_path_length);
	if (block_on_recv(socket_fd, (void *) received_json_path, json_path_length) != json_path_length) 
		{
			return -1;
		}

	// Receive the length of the filename
	int hdf5_path_length = 0;
	if (block_on_recv(socket_fd, (void *) &hdf5_path_length, sizeof(int)) != sizeof(int)) 
		{
			return -1;
		}

	// Receive the filename
	char * received_hdf5_path = malloc(sizeof(char) * hdf5_path_length);
	if (block_on_recv(socket_fd, (void *) received_hdf5_path, hdf5_path_length) != hdf5_path_length) 
		{
			return -1;
		}	

	char * cwd = malloc(100 * sizeof(char));
	getcwd(cwd, 99);
	cwd[99] = 0; 

	// Sanitize the filename by purging anything but alphanumeric characters (regex \w)
	char * cleaned_json_path = g_regex_replace(path_cleaning_pattern, received_json_path, strlen(received_json_path), 0, "", 0, NULL);
	char * final_json_path = g_strdup_printf ("%s/%s.json", cwd, cleaned_json_path);

	char * cleaned_hdf5_path = g_regex_replace(path_cleaning_pattern, received_hdf5_path, strlen(received_hdf5_path), 0, "", 0, NULL);
	char * final_hdf5_path = g_strdup_printf ("%s/%s.hdf5", cwd, cleaned_hdf5_path);

	locking_hdf5_load_params(global_param_list, final_json_path, final_hdf5_path);

	g_free(final_json_path);
	g_free(cleaned_json_path);
	g_free(received_json_path);
		
	return 0;
}



