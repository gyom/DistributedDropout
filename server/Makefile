CC = gcc
#CFLAGS = -fPIC -Wall -std=gnu99 -Wunused -Wunknown-pragmas -Wextra -Wfloat-equal -g
CFLAGS = -fPIC -Wall -std=gnu99 -Wunused -Wunknown-pragmas -Wextra -Wfloat-equal


COMPILING_LOCATION = umontreal
#COMPILING_LOCATION = home_linux_guillaume
#COMPILING_LOCATION = aws_linux
#COMPILING_LOCATION = OSX

ifeq ($(COMPILING_LOCATION),umontreal)

# for pthread.h
export EXTRA_INCLUDE_PATH=/usr/include/
# for jansson and hdf5
export EXTRA_LIB_PATH_JANSSON=umontreal_dependencies/jansson/lib

#export EXTRA_LIB_PATH_HDF5=umontreal_dependencies/hdf5/lib
export EXTRA_LIB_PATH_HDF5=/usr/lib64/libhdf5.so.6.0.4
# /usr/lib64/libhdf5.so.6.0.4

export C_INCLUDE_PATH=umontreal_dependencies/jansson/include:/usr/include

# not tested
#export GLIB=/usr/lib64
export GLIB_INCLUDE_0=/usr/lib64/glib-2.0/include
export GLIB_INCLUDE_1=/usr/include/glib-2.0

else ifeq ($(COMPILING_LOCATION),aws_linux)

# for pthread.h
export EXTRA_INCLUDE_PATH=/usr/include/

# `pkg-config --cflags-only-I glib-2.0`
export GLIB_INCLUDE_0=/usr/lib/x86_64-linux-gnu/glib-2.0/include
export GLIB_INCLUDE_1=/usr/include/glib-2.0

else ifeq ($(COMPILING_LOCATION),OSX)

# for pthread.h
export EXTRA_INCLUDE_PATH=/usr/include/

# for jansson and hdf5
ifeq ($(HDF5),"")
export HDF5="./usr/local/Cellar/hdf5/1.8.14/"
endif

export EXTRA_LIB_PATH_JANSSON=osx_dependencies/jansson/lib
export EXTRA_LIB_PATH_HDF5=$(HDF5)/lib:/usr/lib64/
export C_INCLUDE_PATH=osx_dependencies/jansson/include:$(HDF5)/include

export GLIB=/usr/local/Cellar/glib/2.44.0/
export GLIB_INCLUDE_0=$(GLIB)lib/glib-2.0/include/
export GLIB_INCLUDE_1=$(GLIB)include/glib-2.0/

else ifeq ($(COMPILING_LOCATION), home_linux_guillaume)

# for pthread.h
export EXTRA_INCLUDE_PATH=/usr/include/

# `pkg-config --cflags-only-I glib-2.0`
export GLIB_INCLUDE_0=/usr/lib/x86_64-linux-gnu/glib-2.0/include
export GLIB_INCLUDE_1=/usr/include/glib-2.0


endif

EXTERNAL_LIB_INCLUDE_STUFF = -I. -I$(C_INCLUDE_PATH) -I$(EXTRA_INCLUDE_PATH) -I$(GLIB_INCLUDE_0) -I$(GLIB_INCLUDE_1)
$(info $$LINKING_FLAGS is [${LINKING_FLAGS}])
LINKING_FLAGS = -L${EXTRA_LIB_PATH_JANSSON} -L${EXTRA_LIB_PATH_HDF5}  -lm -lpthread -ljansson -lhdf5

all: main

clean:
	rm -v *.o *.so; rm ../bin/server 

common.o: common.c
	$(CC) $(CFLAGS) $(EXTERNAL_LIB_INCLUDE_STUFF) -c -o common.o common.c

server_hdf5_io.o: server_hdf5_io.c common.h params.h
	$(CC) $(CFLAGS) $(EXTERNAL_LIB_INCLUDE_STUFF) -c -o server_hdf5_io.o server_hdf5_io.c

params.o: params.c common.h handler.h params.h template_commit_slice_to_parameter.c
	$(CC) $(CFLAGS) $(EXTERNAL_LIB_INCLUDE_STUFF) -c -o params.o params.c

handler.o: handler.c common.h handler.h params.h
	$(CC) $(CFLAGS) $(EXTERNAL_LIB_INCLUDE_STUFF) -c -o handler.o handler.c

server_handler.o: server_handler.c common.h handler.h params.h
	$(CC) $(CFLAGS) $(EXTERNAL_LIB_INCLUDE_STUFF) -c -o server_handler.o server_handler.c

main.o:	main.c common.h handler.h params.h server_handler.h
	$(CC) $(CFLAGS) $(EXTERNAL_LIB_INCLUDE_STUFF) -c -o main.o main.c

main: main.o common.o handler.o params.o server_handler.o server_hdf5_io.o
	$(CC) $(CFLAGS) $(EXTERNAL_LIB_INCLUDE_STUFF) main.o server_handler.o server_hdf5_io.o handler.o params.o common.o -o ../bin/server $(LINKING_FLAGS) `pkg-config --cflags --libs glib-2.0` -Wl,-rpath,$(EXTRA_LIB_PATH_HDF5)


