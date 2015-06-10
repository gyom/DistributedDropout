int extract_slice_from_param(param_t * matched_param, msg_param_t * msg);		
int commit_slice_to_param(param_t * matched_param, msg_param_t * msg);
param_t * get_matching_param_entry(param_t * global_param_list, char * name);

int read_MSG_HEADER(msg_header_t * header, int socket_fd);
int read_MSG_PULL_PARAM(msg_param_t * msg, int socket_fd);
int read_MSG_PUSH_PARAM(msg_param_t * msg, int socket_fd);
int respond_MSG_PULL_PARAM(msg_param_t * msg, int socket_fd);
int respond_MSG_LIST_ALL_PARAMS_DESC(param_t * global_param_list, int socket_fd);
int block_on_recv(int socket_fd, void * buffer, int len);
int read_MSG_TYPE_LOAD_ALL_FROM_HDF5(param_t ** global_param_list, int socket_fd);
int read_MSG_TYPE_SAVE_ALL_TO_HDF5(param_t * global_param_list, int socket_fd);
void clean_msg_header(msg_header_t * header);
void clean_msg_param(msg_param_t * msg);

void * server_side_handler(void * conn);

/*
None of those methods have to be shared with the outside world.

int extract_slice_from_param(param_t * matched_param, msg_param_t * msg);		
int commit_slice_to_param(param_t * matched_param, msg_param_t * msg);
param_t * get_matching_param_entry(param_t * global_param_list, char * name);

int read_MSG_HEADER(msg_header_t * header, int socket_fd);
int read_MSG_PULL_PARAM(msg_param_t * msg, int socket_fd);
int read_MSG_PUSH_PARAM(msg_param_t * msg, int socket_fd);
int respond_MSG_PULL_PARAM(msg_param_t * msg);
*/

