# Easy to edit configure script for athena++ 
python3 configure.py\
			 --prob magshell \
			 --flux hllc \
			 --coord cartesian \
			 -hdf5 \
                         -exp \
                         --ns 3 \
                         -mpi \
                         --cxx icc \
                         --ccmd /nas/longleaf/apps-dogwood/hdf5/1.10.2/openmpi/bin/h5pcc \
                         --cflag="DH5_HAVE_PARALLEL -std=c++11" 
                         #--lib /usr/mpi/intel/openmpi-2.0.3/lib64
			 #--include /usr/include/hdf5/openmpi/ \
#			 --hdf5_path=/usr/lib/x86_64-linux-gnu/
