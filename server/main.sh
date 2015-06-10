export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:umontreal_dependencies/hdf5/lib 
#small hack to be able to run from both the client folders and the server folder directly
cd ../server
./main $@
