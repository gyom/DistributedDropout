
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* define _BSD_SOCKLEN_T_ in order to define socklen_t on darwin */
# define _BSD_SOCKLEN_T_
/* one more header to be included on OSX */
#include <netinet/in.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdbool.h>

// for htons
#include <arpa/inet.h>
#include <stdint.h>

//#include <time.h>
#include <pthread.h>

#include "common.h"
#include "handler.h"
#include "params.h"

#include <signal.h>
#include <assert.h>
#include <glib.h>




void addThreadToList(GList ** threads, pthread_t id) {
    thread_liveness_t * thread = malloc(sizeof(thread_liveness_t));
    thread->thread_id = id;
    thread->is_alive = true;
    *threads = g_list_prepend(*threads, thread);
}

void safeAddThreadToList(GList ** threads, pthread_t id, pthread_mutex_t * mutex) {
    pthread_mutex_lock(mutex);
    thread_liveness_t * thread = malloc(sizeof(thread_liveness_t));
    thread->thread_id = id;
    thread->is_alive = true;
    *threads = g_list_append(*threads, thread);
    pthread_mutex_unlock(mutex);
}

void safeJoinAllThreadsFromList(GList ** threads, pthread_mutex_t * mutex) {
    pthread_mutex_lock(mutex);
    GList * node = *threads;
    while(node) {
        GList * next = node->next;
        thread_liveness_t * data = node->data;
        pthread_join(data->thread_id, NULL);
        free(data);
        *threads = g_list_delete_link(*threads, node);
        node = next;
    }
    pthread_mutex_unlock(mutex);
}


int setup_server_and_run(
    param_t * global_param_list,
    uint16_t port,
    int max_nbr_clients,
    int max_memory_alloc_MB,
    gboolean verbose ) {

    /* the arguments to `setup_server_and_run` are basically just the
       global parameters read from main's argv */

    puts("Entered setup_server_and_run.\n");

    int socket_desc;
    struct sockaddr_in server_address;
         
    //Prepare the sockaddr_in structure
    memset (&server_address, 0, sizeof(server_address));        
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons( port );

    // create socket for server
    socket_desc = socket(AF_INET , SOCK_STREAM , 0);
    if (socket_desc == -1)
    {
        printf("main.c - Controller thread: Could not create socket.");
    }


    // bind
    if( bind(socket_desc, (struct sockaddr *)&server_address , sizeof(server_address)) < 0)
    {
        puts("main.c - Controller thread: Bind failed.");
        return -1;
    }
    puts("main.c - Controller thread: Bind done.");

    //listen to incoming connections (pick a backlog of 10)
    listen(socket_desc, 10);


    GList * threads = NULL;
    int threads_max = 300;
    int threads_size = 0;

    while (true) {
        // Check if we can accept a new connection. If not, we will loop until a thread can be removed.
        if(threads_size < threads_max) 
        {
            client_conn_t * ncc_ptr = (client_conn_t *)malloc(sizeof(client_conn_t));
            memset(ncc_ptr, 0, sizeof(client_conn_t));  

            // accept and incoming connection
            puts("main.c - Controller thread: Waiting for incoming connections...");
            ncc_ptr->address_length = sizeof(server_address);
            ncc_ptr->socket_fd = accept(socket_desc, (struct sockaddr *)&(ncc_ptr->client), (socklen_t*)&(ncc_ptr->address_length));

            if (ncc_ptr->socket_fd<0)
            {
                perror("accept failed");
                ncc_ptr = NULL;
                continue;
            }
            puts("main.c - Controller thread: Connection accepted");

            /* http://stackoverflow.com/questions/2876024/linux-is-there-a-read-or-recv-from-socket-with-timeout */
            //struct timeval tv;
            //tv.tv_sec = 180;  /* 180 Secs Timeout */
            //tv.tv_usec = 0;  // Not init'ing this can cause strange errors
            //if (setsockopt(ncc_ptr->socket_fd, SOL_SOCKET, SO_RCVTIMEO, (char *)&tv,sizeof(struct timeval)) < 0) {
            //    perror("setsockopt failed\n");
            //}

            //SetSocketBlockingEnabled(ncc_ptr->socket_fd, true);

            ncc_ptr->want_thread_to_stop = false;
            ncc_ptr->global_param_list = global_param_list;
            ncc_ptr->max_data_length_bytes = 128*1024*1024; // go for 128 MB for now

            pthread_create(&(ncc_ptr->thread_id), NULL, &server_side_handler, ncc_ptr);


            addThreadToList(&threads, ncc_ptr->thread_id);
            ncc_ptr->is_alive = &((thread_liveness_t * )threads->data)->is_alive;
            ++threads_size;


            printf("main.c - Controller thread: thread_liveness_t created. thread_liveness_t list size : %d\n", threads_size);
            /* Register ncc_ptr in some kind of linked list where we keep all the connections.
               Then get rid of that pointer here to make it clear that we're allocating
               a new one every time we are getting ready to accept a connection.
            */
            ncc_ptr = NULL;

            
            
        } else {
            puts("main.c - Controller thread: thread_liveness_t liveness queue is full.");
        }

        // Check if there are any dead threads
        GList * node = threads;
        if(node == NULL) {
            puts("wtf");
        }
        while(node) 
        {
            thread_liveness_t * data = node->data;
            printf(" >>> Checking liveness of thread #%lu: value is %s\n", (size_t)data->thread_id, data->is_alive?"true":"false");
            GList * next = node->next;
            
            if(!data->is_alive) 
            {
                // Join an eventual dead thread
                printf("main.c - Controller thread: trying to join thread #%lu\n", (size_t)data->thread_id);
                pthread_join(data->thread_id, NULL);
                // Remove its node. Nodes know their parent list.
                free(data);
                threads = g_list_delete_link(threads, node);
                printf("main.c - Controller thread: joined thread #%lu\n", (size_t)data->thread_id);
            }
            node = next;
        }
    }

    // Kill all threads, free the nodes. The list object itself is stack based (automatic).

    /*
    The code used to be
        g_list_free_full(threads, NULL);
    but this wasn't supported in glib-2.0 installed at umontreal.
    See this reference for a substitution :
        https://github.com/fontforge/fontforge/issues/995
        https://github.com/fontforge/fontforge/pull/1121/files
    Guillaume (me) specified this reference because he did the
    substitution more or less blindly.
    */
    g_list_foreach( threads, NULL, NULL );
    g_list_free( threads );

    threads = NULL;
}


