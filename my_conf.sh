# Easy to edit configure script for athena++ 
python3 configure.py\
			 --prob advectiontest\
			 --flux bgk2n\
			 --coord cartesian\
			 --nf=2 \
			 --nghost=2 \
			 -hdf5 \
                         -mpi \
                         --cflag="-DH5_HAVE_PARALLEL -std=c++11" \
                         --ccmd /nas/longleaf/apps-dogwood/hdf5/1.10.2/openmpi/bin/h5pcc \
			 --include /srv/analysis/local/hdf5/include \
			 --hdf5_path=/srv/analysis/local/hdf5/ \
			 -debug
