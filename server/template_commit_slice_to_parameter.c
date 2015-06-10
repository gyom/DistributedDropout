
#include <pthread.h>

#include "common.h"

/*
    Note that you'll have problem when it comes to float16
    because it's not supported in C.
    http://stackoverflow.com/questions/3026441/float32-to-float16
*/

/*  This is one version of out many.
    It's not clear that we want to spend time
    to copy/paste (or something more clever)
    so as to have all the variations in types
    as well as the variations where `nbr_subelements==1`
    allows us to skip one level of looping.
   */

int commit_slice_to_param_float32_to_float32(
	pthread_mutex_t mutex,
	slice_t * slice_ptr,
	float * src,
	float * dst,
	int nbr_subelements,
	float alpha, float beta ) {

	//printf("Received call to commit_slice_to_param_float32_to_float32 with (alpha=%f, beta=%f).\n", alpha, beta);

	// acquire mutex
	pthread_mutex_lock(&mutex);

	// TODO : decide whether you even want the possibility of
	//        specifying 0 as dimensions. Why not use 1 at
	//        every dimension that we don't care about ?

	/* replace dimensions 0 with the value 1
	   so that we have cleaner loops */
	
	/*
	int S[2];
	S[0] = slice_ptr->S[0];
	S[1] = slice_ptr->S[1];
	S[0] = 1 <= S[0] ? S[0] : 1;
	S[1] = 1 <= S[1] ? S[1] : 1;

	int D[2];
	D[0] = slice_ptr->D[0];
	D[1] = slice_ptr->D[1];
	D[0] = 1 <= D[0] ? D[0] : 1;
	D[1] = 1 <= D[1] ? D[1] : 1;
	*/

	//printf("In commit_slice_to_param_float32_to_float32, nbr_subelements : %d\n", nbr_subelements);

	for (int i = 0; i < slice_ptr->S[0]; i++) {
		int indi = slice_ptr->indices[0][i];

		for (int j = 0; j < slice_ptr->S[1]; j++) {
			int indj = slice_ptr->indices[1][j];

			//printf("(i, indi) : (%d, %d)    ", i, indi);
			//printf("(j, indj) : (%d, %d)\n", j, indj);

			int offset_dst = (indi*slice_ptr->D[1] + indj)*nbr_subelements;
			int offset_src = (i*slice_ptr->S[1] + j)*nbr_subelements;

			for (int k=0; k < nbr_subelements; k++) {
				dst[offset_dst + k] = (float)(alpha * dst[offset_dst + k] + beta * src[offset_src + k]);
			}
		}
	}

	// release mutex
	pthread_mutex_unlock(&mutex);

	return 0;
}



int extract_slice_to_param_float32_to_float32(
	pthread_mutex_t mutex,
	slice_t * slice_ptr,
	float * src,
	float * dst,
	int nbr_subelements) {

	// acquire mutex
	// (because we don't want someone to write to this param while we read its value)
	pthread_mutex_lock(&mutex);

	//printf("In extract_slice_to_param_float32_to_float32, nbr_subelements : %d\n", nbr_subelements);

	for (int i = 0; i < slice_ptr->S[0]; i++) {
		int indi = slice_ptr->indices[0][i];

		for (int j = 0; j < slice_ptr->S[1]; j++) {
			int indj = slice_ptr->indices[1][j];

			//printf("(i, indi) : (%d, %d)    ", i, indi);
			//printf("(j, indj) : (%d, %d)\n", j, indj);


			float * sub_dst = &dst[(i * slice_ptr->S[1] + j) * nbr_subelements];
			float * sub_src = &src[(indi * slice_ptr->D[1] + indj) * nbr_subelements];
			// might have a branch depending on if it's worth doing subcalls to mempcy
			for (int s = 0; s < nbr_subelements; s++) {
				sub_dst[s] = sub_src[s];
			}
		}
	}

	// release mutex
	pthread_mutex_unlock(&mutex);

	return 0;
}



