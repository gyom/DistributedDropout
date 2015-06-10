
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>
#include <pthread.h>
#include <signal.h>

// define _BSD_SOCKLEN_T_ in order to define socklen_t on darwin
# define _BSD_SOCKLEN_T_
// one more header to be included on OSX
#include <netinet/in.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdbool.h>

// for htons
#include <arpa/inet.h>

#include <glib.h>
#include <hdf5.h>

#include "common.h"
#include "handler.h"
#include "params.h"
#include "server_handler.h"

#include <signal.h>
#include <assert.h>
#include <glib.h>
#include <glib/gprintf.h>

static param_t * global_param_list;

/* boilerplate code from https://developer.gnome.org/glib/stable/glib-Commandline-option-parser.html */
static gint port = 0;
static gint max_nbr_clients = 0;
static gint max_memory_alloc_MB = 0;
static gboolean verbose = FALSE;
static gchar * model_params_data_input = NULL;
static gchar * model_params_desc = NULL;
//static gchar * model_params_data_default_output = NULL;
// optional : might want to add something for the network interface on which we listen

static GOptionEntry entries[] =
{
    {"port",                    0,   0, G_OPTION_ARG_INT,    &port,                    "port on which the server listens to", NULL },   
    {"max_nbr_clients",         0,   0, G_OPTION_ARG_INT,    &max_nbr_clients,         "maximum number of connected clients at the same time", NULL },
    {"max_memory_alloc_MB",     0,   0, G_OPTION_ARG_INT,    &max_memory_alloc_MB,     "maximum amount of memory allocated before refusing client connections", NULL },
    {"verbose",                 'v', 0, G_OPTION_ARG_NONE,   &verbose,                 "verboose mode", NULL },
    {"model_params_data_input", 'i', 0, G_OPTION_ARG_STRING, &model_params_data_input, "model parameters data (hdf5 filename)", NULL },
    {"model_params_desc",       'd', 0, G_OPTION_ARG_STRING, &model_params_desc,       "model parameters structure description (json filename)", NULL},  
    //  {"model_params_data_default_output", 'i', 0, G_OPTION_ARG_STRING, &model_params_data_input, "model parameters data (hdf5 filename)", NULL },
    { NULL }
};

int main(int argc, char *argv[]) {

    GError *error = NULL;
    GOptionContext *context;

    context = g_option_context_new("- Parameter server for the NIPS2015 Distributed Dropout Training project.");
    g_option_context_add_main_entries(context, entries, NULL);
    //g_option_context_add_group(context, gtk_get_option_group(TRUE));
    if (!g_option_context_parse(context, &argc, &argv, &error)) {
        g_printf("option parsing failed: %s\n", error->message);
        exit(1);
        }

    if (model_params_desc == NULL) {
        g_printf("Fatal error : Missing `model_params_desc` argument. \n");
        g_printf("              We cannot start the server without a JSON file describing the model parameters.\n");
        exit(1);
    }

    if (port < 1024 || 65535 <= port) {
        g_printf("Fatal error : Requires a port number between 1024 and 65535.\n");
        exit(1);
    }

    if (max_nbr_clients == 0 && max_memory_alloc_MB == 0) {
        g_printf("Warning : You're accepting an unlimited number of clients and you have not limited the total memory usage.\n");
        g_printf("          This can lead to the server over-allocating memory.\n");
    }

    if (model_params_data_input == NULL) {
        g_printf("Notice : Missing `model_params_data_input` argument.\n");
        g_printf("         Will initialize the server parameters with zeros.\n");
    }

    global_param_list = read_params_from_json_file(model_params_desc);
    if ( global_param_list == NULL ) {
        g_printf("Fatal error : Failed to parse the `model_params_desc` in '%s'.\n", model_params_desc);
        exit(1);        
    }

    //global_param_list = read_params_from_json_file("debug_files/melanie_mnist_params_desc_02.json");
    //global_param_list = read_params_from_json_file("simple_params_desc.json");
    //global_param_list = make_test_param_list();

    /*  Since we're reading the arguments into the global memory,
        we might as well not pass them to the `setup_server_and_run` method.
        TODO : Discuss putting the handler and everything in another file.
    */

    setup_server_and_run(global_param_list, (uint16_t)port, (int)max_nbr_clients, (int)max_memory_alloc_MB, verbose);

    return 0;
}


/*

    Example usage :

        ./main --port=5005 --model_params_desc=debug_files/melanie_mnist_params_desc_02.json

*/





