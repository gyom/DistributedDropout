#include <string.h>
#include <stdlib.h>
#include <jansson.h>

#include "params.h"
#include "common.h"
#include "template_commit_slice_to_parameter.c"


param_t * new_param_from_json_t(json_t * e);
param_t * new_param_float32(char * name, int shape[4], int kind);


int param_kind_str_to_kind_int(const char * kind_str) {
	if (strcmp(kind_str, "FULLY_CONNECTED_WEIGHTS") == 0) { return FULLY_CONNECTED_WEIGHTS; }
	if (strcmp(kind_str, "FULLY_CONNECTED_BIASES") == 0) { return FULLY_CONNECTED_BIASES; }
	if (strcmp(kind_str, "CONV_FILTER_WEIGHTS") == 0)    { return CONV_FILTER_WEIGHTS; }
	if (strcmp(kind_str, "CONV_FILTER_BIASES") == 0)     { return CONV_FILTER_BIASES; }
	thread_printf("params.c: Unrecognized kind in param_kind_str_to_kind_int.\n");
	// not fun, but nothing fatal since we don't even use those values
	return 0;	
}

char * param_kind_int_to_kind_str(int kind_int) {
	/* Note that you should not attempt to free(...) those values once you're done with them. */
	if (kind_int == FULLY_CONNECTED_WEIGHTS) { return "FULLY_CONNECTED_WEIGHTS"; }
	if (kind_int == FULLY_CONNECTED_BIASES)  { return "FULLY_CONNECTED_BIASES"; }
	if (kind_int == CONV_FILTER_WEIGHTS)     { return "CONV_FILTER_WEIGHTS"; }
	if (kind_int == CONV_FILTER_BIASES)      { return "CONV_FILTER_BIASES"; }
	thread_printf("params.c: Unrecognized kind in param_kind_int_to_kind_str.\n");
	// not fun, but nothing fatal since we don't even use those values
	return "";
}

param_t * read_params_from_json_file(const char * filename) {
	json_t * contents;
    json_error_t error;

    contents = json_load_file(filename, 0, &error);
    if (!contents) {
    	fprintf(stderr, "Error in libjansson while reading file %s.\n", filename);
    	fprintf(stderr, "text : %s\n", error.text);
    	fprintf(stderr, "source : %s\n", error.source);
    	fprintf(stderr, "line : %d\n", error.line);
    	fprintf(stderr, "column : %d\n", error.column);
    	fprintf(stderr, "position : %d\n", error.position);
    	return NULL;
    }

    if (!json_is_array(contents)) {
    	fprintf(stderr, "Error. Expected json contained in file %s to be a list of elements.\n", filename);
    	return NULL;
    }

    param_t * head = NULL;
    param_t * previous_param = NULL;
    param_t * current_param = NULL;

    for (size_t i = 0; i < json_array_size(contents); i++) {
   		thread_printf("params.c: %d out of %zu\n", i, json_array_size(contents));

    	json_t * e = json_array_get(contents, i);
    	if (e == NULL) {
    		fprintf(stderr, "Error. The element %lu out of %zu was supposed to be valid, but it's NULL.\n", i, json_array_size(contents));
    		return NULL;
    	}

    	current_param = new_param_from_json_t(e);
    	if (current_param == NULL) {
    		fprintf(stderr, "Error parsing element %lu out of %zu.\n", i, json_array_size(contents));
    		return NULL;
    	}

    	if (head == NULL) {
    		// set the head to the first element
    		head = current_param;
    	}

    	if (previous_param) {
    		previous_param->next = current_param;
    	}

    	previous_param = current_param;
    }

    // TODO : Should we free(contents) at this point or do the jansson equivalent ?

    return head;
}

param_t * new_param_from_json_t(json_t * e) {
	if ( !json_is_object(e) ) {
		return NULL;
	}

	json_t * namej = json_object_get(e, "name");
	json_t * kindj = json_object_get(e, "kind");
	json_t * shapej = json_object_get(e, "shape");

	char * name = NULL;
	if (!namej || !json_is_string(namej)) {
		fprintf(stderr, "Invalid value for name.\n");
	 	return NULL;
	} else {
		// dropping the `const` by casting explicitly. don't ever free(name)
		name = (char *)json_string_value(namej);
	}

	int kind;
	if (kindj && json_is_integer(kindj)) {
		kind = json_integer_value(kindj);
	} else if (kindj && json_is_string(kindj)) {
		const char * kind_str = json_string_value(kindj);
		kind = param_kind_str_to_kind_int((char *)kind_str);
	} else {
		fprintf(stderr, "Invalid value for kind.\n");
		return NULL;
	}

	/* 	the default value is `1`.
		When less than 4 values are specified,
		the behavior depends on what kind of `kind`
		we're dealing with.
	*/
	int shape[4] = {1, 1, 1, 1};
	if (shapej && json_is_array(shapej) && json_array_size(shapej) == 4) 
	{
		/* don't bother checking anything; just transcribe the 4 values */
		for (int i = 0; i < 4; ++i) 
		{
			json_t * s = json_array_get(shapej, i);
			if (!json_is_integer(s)) 
			{
				fprintf(stderr, "Got shape value that wasn't an integer.\n");
				return NULL;
			}

			shape[i] = json_integer_value(s);
		}

	} else if ( shapej && json_is_array(shapej) && json_array_size(shapej) == 3) {
		if (kind != CONV_FILTER_BIASES) 
		{
			fprintf(stderr, "Error. The only shape that can have 3 entries is a CONV_FILTER_BIASES, but instead we got 3 values for a %s.\n", param_kind_int_to_kind_str(kind));
			return NULL;
		}

		for (int i=0; i < 3; i++) 
		{
			json_t * s = json_array_get(shapej, i);
			if ( !json_is_integer(s) ) 
			{
				fprintf(stderr, "Got shape value that wasn't an integer.\n");
				return NULL;
			}
		
			// notice the change in the shape index here
			shape[1+i] = json_integer_value(s);
		}

	} else if ( shapej && json_is_array(shapej) && json_array_size(shapej) == 2) 
	{
		if (kind != FULLY_CONNECTED_WEIGHTS) 
		{
			fprintf(stderr, "Error. The only shape that can have 2 entries is a FULLY_CONNECTED_WEIGHTS, but instead we got 2 values for a %s.\n", param_kind_int_to_kind_str(kind));
			return NULL;
		}

		for (int i=0; i < 2; i++) 
		{
			json_t * s = json_array_get(shapej, i);
			if ( !json_is_integer(s) ) 
			{
				fprintf(stderr, "Got shape value that wasn't an integer.\n");
				return NULL;
			}

			// notice the change in the shape index here			
			shape[i] = json_integer_value(s);
		}

	} else if ( shapej && json_is_array(shapej) && json_array_size(shapej) == 1) 
	{

		if (kind != FULLY_CONNECTED_BIASES) 
		{
			fprintf(stderr, "Error. The only shape that can have 1 entries is a FULLY_CONNECTED_BIASES, but instead we got 1 values for a %s.\n", param_kind_int_to_kind_str(kind));
			return NULL;
		}

		for (int i=0; i < 1; i++) {
			json_t * s = json_array_get(shapej, i);
			if ( !json_is_integer(s) ) 
			{
				fprintf(stderr, "Got shape value that wasn't an integer.\n");
				return NULL;
			}

			// notice the change in the shape index here
			shape[1+i] = json_integer_value(s);
		}

	} else 
	{
		fprintf(stderr, "Invalid value for shape.\n");
		return NULL;
	}

	param_t * res = new_param_float32(name, shape, kind);


	return res;
}

param_t * new_param_float32(char * name, int shape[4], int kind) {

	param_t * p0 = (param_t *)malloc(sizeof(param_t));
	memset(p0->name, '\0', PARAM_NAME_LENGTH);
	strcpy(p0->name, name);
	p0->shape[0] = shape[0];
	p0->shape[1] = shape[1];
	p0->shape[2] = shape[2];
	p0->shape[3] = shape[3];
	p0->kind = kind;
	//p0->mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_mutex_init(&p0->mutex, NULL);
	p0->data_length_bytes = p0->shape[0] * p0->shape[1] * p0->shape[2] * p0->shape[3] * sizeof(float);
	p0->data = malloc(p0->data_length_bytes);
	memset(p0->data, 0, p0->data_length_bytes);
	p0->dtype = DTYPE_FLOAT32;
	p0->next = NULL;

	for (size_t i = 0u, e = shape[0] * shape[1] * shape[2] * shape[3]; i < e; ++i) {
		*(float*)(p0->data + i * sizeof(float)) = 0.f;
	}

	return p0;
}

void free_param(param_t * p) {
	g_free(p->data);
	p->data = NULL;
	// TODO : anything required to clean up the mutex ?
}

json_t * encode_param_to_json_t(param_t * p) {
	json_t * res = json_object();
	json_object_set_new(res, "name", json_string((const char *)p->name));
	json_object_set_new(res, "kind", json_string((const char *)param_kind_int_to_kind_str(p->kind)));
	json_object_set_new(res, "shape", json_pack("[iiii]", p->shape[0], p->shape[1], p->shape[2], p->shape[3]));
	return res;
}

json_t * encode_list_param_to_json_t(param_t * head) 
{
	json_t * res = json_array();
	param_t * p = head;
	while (p != NULL) 
	{
		//printf("at parameter %s\n", p->name);
		if (json_array_append_new(res, encode_param_to_json_t(p)) == -1) 
		{
			fprintf(stderr, "Error. Having problems in encode_list_param_to_json_t to add to array.\n");
			return NULL;
		}
		p = p->next;
	}

	return res;
}

/* JUNK : We can get rid of this eventually. */
param_t * make_test_param_list() 
{
	/*
	[
		{"name" : "layer1_weights",
    	 "kind" : "CONV_FILTER_WEIGHTS",
	     "shape" : [32, 3, 7, 7]},
	    {"name" : "layer1_biases",
	     "kind" : "CONV_FILTER_BIASES",
    	 "shape" : [32, 1, 22, 22]}
    ]
	*/
	param_t * p0 = (param_t *)malloc(sizeof(param_t));
	param_t * p1 = (param_t *)malloc(sizeof(param_t));

	memset(p0->name, '\0', PARAM_NAME_LENGTH);
	strcpy(p0->name, "layer1_weights");
	p0->shape[0] = 32;
	p0->shape[1] = 3;
	p0->shape[2] = 7;
	p0->shape[3] = 7;
	p0->kind = CONV_FILTER_WEIGHTS;
	//p0->mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_mutex_init(&p0->mutex, NULL);
	p0->data_length_bytes = p0->shape[0] * p0->shape[1] * p0->shape[2] * p0->shape[3] * sizeof(float);
	p0->data = malloc(p0->data_length_bytes);
	memset(p0->data, 0, p0->data_length_bytes);
	p0->dtype = DTYPE_FLOAT32;
	p0->next = NULL;

	/* populate with something more fun */
	for (size_t t = 0; t < p0->data_length_bytes / sizeof(float); ++t) 
	{
		((float*)(p0->data))[t] = t;
	}

	memset(p1->name, '\0', PARAM_NAME_LENGTH);
	strcpy(p1->name, "layer1_biases");
	p1->shape[0] = 32;
	p1->shape[1] = 1;
	p1->shape[2] = 7;
	p1->shape[3] = 7;
	p1->kind = CONV_FILTER_BIASES;
	//p0->mutex = PTHREAD_MUTEX_INITIALIZER;	
	pthread_mutex_init(&p1->mutex, NULL);
	p1->data_length_bytes = p1->shape[0] * p1->shape[1] * p1->shape[2] * p1->shape[3] * sizeof(float);
	p1->data = malloc(p1->data_length_bytes);
	memset(p1->data, 0, p1->data_length_bytes);
	p1->dtype = DTYPE_FLOAT32;
	p1->next = NULL;

	/* populate with something more fun */
	for (size_t t = 0; t < p1->data_length_bytes / sizeof(float); ++t) 
	{
		((float*)(p1->data))[t] = t;
	}

	p0->next = p1;
	return p0;
}

