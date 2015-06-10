
#ifndef __PARAMS_H__
#define __PARAMS_H__

#include "common.h"
#include <jansson.h>

int param_kind_str_to_kind_int(const char * kind_str);
char * param_kind_int_to_kind_str(int kind_int);

json_t * encode_param_to_json_t(param_t * p);
json_t * encode_list_param_to_json_t(param_t * head);
param_t * read_params_from_json_file(const char * filename);

void free_param(param_t * p);


int commit_slice_to_param_float32_to_float32(
	pthread_mutex_t mutex,
	slice_t * slice_ptr,
	float * src,
	float * dst,
	int nbr_subelements,
	float alpha, float beta );

int extract_slice_to_param_float32_to_float32(
	pthread_mutex_t mutex,
	slice_t * slice_ptr,
	float * src,
	float * dst,
	int nbr_subelements);

param_t * make_test_param_list();

#endif