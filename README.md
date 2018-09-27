# Tensorflow_With_MPI_on_Nvidia_TX2_Wheel

## Introduction
Wheel for Tensorflow with MPI support on Nvidia Jetson TX2.

##System Info
### JetPack 3.3, TensorFlow 1.10
2018 9/27

1. cuDNN v7.1.5
2. CUDA 9.0		
3. Python 3.5
4. TF with MPI support: Yes

## Troubleshooting
### 1. When building tensrflow: '_NamespacePath' object has no attribute 'sort'

The line in pkg_resources/_init_.py that reads:
```
orig_path.sort(key=position_in_sys_path)
```

Should be:

```
orig_path = sorted(orig_path, key=position_in_sys_path)
```

If you cannot change the source code, try:
```
sudo pip3 install -U setuptools
```

### 2. When running MPI: unable to open /usr/lib/openmpi/lib/openmpi/mca_shmem_posix, returned value -1 instead of opal_success

Run the following command:

```
sudo apt-get remove mpi4py
```

Then install the Open MPI headers (the next step involves building mpi4py) and pip:

```
sudo apt-get install libopenmpi-dev python-pip
sudo pip install mpi4py
```

Finally, set LD_PRELOAD environmental variable to the location of libmpi.so 
```
export LD_PRELOAD=/usr/local/openmpi/lib/libmpi.so
```

