
#ifndef __SERVER_HANDLER_H__
#define __SERVER_HANDLER_H__

int setup_server_and_run(
    param_t * global_param_list,
    uint16_t port,
    int max_nbr_clients,
    int max_memory_alloc_MB,
    gboolean verbose );

#endif