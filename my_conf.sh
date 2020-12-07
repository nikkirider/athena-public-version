# Easy to edit configure script for athena++ 
python3 configure.py\
			 --prob shock_tube\
			 --flux bgk2\
			 --coord cartesian \
			 --nghost=2 \
			 -hdf5 \
			 --include /srv/analysis/local/hdf5/include \
			 --hdf5_path=/srv/analysis/local/hdf5/ \
			 -debug
