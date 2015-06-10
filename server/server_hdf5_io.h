#ifndef __SERVER_HDF5_IO_H__
#define __SERVER_HDF5_IO_H__
#include <hdf5.h>
#include <glib.h>
#include "params.h"


herr_t locking_save_to_hdf5(param_t * head, char * hdf5_path);
herr_t save_to_hdf5(param_t * head, char * hdf5_path);
herr_t locking_save_to_hdf5(param_t * head, char * hdf5_path);
void param_from_hdf5_dataset(hid_t parent_id, param_t * param);
void locking_hdf5_load_params(param_t ** params, const char * path_to_json, const char * path_to_hdf5);


#endif

