#include <hdf5.h>
#include <assert.h>
#include "server_hdf5_io.h"
#include "common.h"

/* Based on
   https://www.hdfgroup.org/HDF5/examples/api18-c.html
   https://www.hdfgroup.org/ftp/HDF5/examples/examples-by-api/hdf5-examples/1_8/C/H5G/h5ex_g_create.c

   https://www.hdfgroup.org/HDF5/Tutor/datatypes.html
   https://www.hdfgroup.org/HDF5/doc1.6/Datasets.html
   https://www.hdfgroup.org/HDF5/doc1.6/RM_H5D.html#Dataset-Create
 */



// Same as save_to_hdf5, but locks a parameter's mutex before reading it.
herr_t locking_save_to_hdf5(param_t * head, char * hdf5_path) {

    /* takes a linked list that starts with `head`
       and serializes all the parameters into one group
       for each parameter
    */
    
    hid_t file, group, space, dset, dcpl;         /* Handles */
    herr_t status;

    /* Create a new file using the default properties. */
    file = H5Fcreate(hdf5_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create a group to store all the model parameters (later groups might be used for other things like logging). */
    group = H5Gcreate(file, "model_params", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t dims[4];
    param_t * p = head; // better name for traveling pointer
    while(p) {
        pthread_mutex_lock (&p->mutex);
        for (int k=0; k < 4; k++) {
            dims[k] = p->shape[k];
        }


        space = H5Screate_simple(4, dims, NULL);

        dcpl = H5Pcreate(H5P_DATASET_CREATE);
        status = H5Pset_layout(dcpl, H5D_CONTIGUOUS);

        /* Create the dataset. */
        dset = H5Dcreate(group, p->name, H5T_NATIVE_FLOAT, space, H5P_DEFAULT, dcpl,
                          H5P_DEFAULT);

        /* we should also store attributes, even though they're not absolutely needed 
         * since we can infer the shape and data type from the hdf5 format
         */
        status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, p->data);

        status = H5Pclose(dcpl);
        status = H5Dclose(dset);
        status = H5Sclose(space);
        pthread_mutex_unlock (&p->mutex);
        p = p->next;
    }

    status = H5Gclose(group);
    status = H5Fclose(file);
    
    // if there was an error, you should have returned earlier than this
    return status;
}


void param_from_hdf5_dataset(hid_t parent_id, param_t * param) {

    hid_t dataset = H5Dopen2(parent_id, param->name, H5P_DEFAULT);
    printf("param_t name: %s\n", param->name);
    H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, param->data);
    H5Dclose(dataset);
}


void locking_hdf5_load_params(param_t ** params, const char * path_to_json, const char * path_to_hdf5) {
    
    *params = read_params_from_json_file(path_to_json);
    param_t * start = *params;

    assert(params && "Failed to read params from json file");
    
    printf("locking_hdf5_load_params: path is '%s'\n", path_to_hdf5);
    hid_t fileID = H5Fopen(path_to_hdf5, H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t rootGroupID = H5Gopen(fileID, "/", H5P_DEFAULT);
    hsize_t numInGrp;
    hid_t main_group = H5Gopen(rootGroupID, "model_params", H5P_DEFAULT);
    H5Gget_num_objs(main_group, &numInGrp);
    printf("number of objects in the root group: %llu\n", numInGrp);
    while(*params) {
        printf("locking_hdf5_load_params : trying param %s.\n", (*params)->name);
        pthread_mutex_lock(&(*params)->mutex);
        param_from_hdf5_dataset(main_group, *params);
        pthread_mutex_unlock(&(*params)->mutex);
        (*params) = (*params)->next;
    }

    // We want *params to be pointing to the first param in the list
    *params = start;
}
