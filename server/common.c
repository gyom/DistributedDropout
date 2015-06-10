#include <common.h>


void thread_printf(char *format, ...)
{	
	printf("Thread #%lu: ", (size_t)pthread_self());

	va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

////////////////////////////////////////////////////////////////
// Logging intended for statistics and more macroscopic book keeping 
// than actual micro management like what would get printed in the console.
// We call it log_info because log was already used .. (for the logarithm function)
////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)

#ifdef __GNU_C__
	#define UNUSED_FUNC __attribute__((unused))
#else
	#define UNUSED_FUNC
#endif

UNUSED_FUNC void log_info(size_t fd, char * str, size_t len) {
	// Is currently a placeholder. We know we will eventually need this though.
	// http://stackoverflow.com/questions/3599160/unused-parameter-warnings-in-c-code
	UNUSED(fd);
	UNUSED(str);
	UNUSED(len);
}