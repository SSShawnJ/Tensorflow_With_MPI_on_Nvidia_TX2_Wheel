# Run Distributed TF using MPI

## Important Notes
1. Execute the script 'reinstall.sh' in ~/Documents, When the tf job completed each time.
2. The location to put the required scripts:
- job_dispatcher.py: working_directory/
- moe.py: working_directory/
- utility.py: working_directory/
- shake_shake.py: working_directory/tensor2tensor/models/
- basic.py: working_directory/tensor2tensor/models/
3. Master device is 44323
4. When running cifar10, you can choose whether to split the computation graph using concolution or branch by setting by_conv flag = True or False.

## 2 Devices
### MNIST
On 44323 and 44324:

1. ```cd ~/workspace/teamnet_mpi/```
2. In basic.py, uncommnet the code under '# 2 devices' and comment the code under '# 4 devices'
3. Do point 2 in important notes.

On 44323:

4. Open a terminal and run: 
- ```mpirun -host master,lanhost -bind-to none -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude l4tbr0,docker0,lo python3 -m mpi4py job_dispatcher.py --worker_hosts=master:44323,lanhost:44324 --task=mnist --device=cpu```
- You can switch device between 'cpu' and 'gpu'

On 44323 and 44324:
5. Do point 1 in important notes.

### CIFAR10
On 44323 and 44324:

1. ```cd ~/workspace/teamnet_mpi/```
2. (If by_conv=True) In shake_shake.py, inside my_conv_2d() function, uncommnet the code under '# 2 devices' and comment the code under '# 4 devices'
3. Do point 2 in important notes.

On 44323:

4. Open a terminal and run: 
- ```mpirun -host master,lanhost -bind-to none -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude l4tbr0,docker0,lo python3 -m mpi4py job_dispatcher.py --worker_hosts=master:44323,lanhost:44324 --task=cifar10 --device=cpu --by_conv=False```
- You can switch device between 'cpu' and 'gpu'

On 44323 and 44324:
5. Do point 1 in important notes.

## 4 Devices
### MNIST
On 44323, 44324, 44327, 44328:

1. ```cd ~/workspace/teamnet_mpi/```
2. In basic.py, uncommnet the code under '# 4 devices' and comment the code under '# 2 devices'
3. Do point 2 in important notes.

On 44323:

4. Open a terminal and run: 
- ```mpirun -host master,lanhost,lanhost2,lanhost3 -bind-to none -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude l4tbr0,docker0,lo python3 -m mpi4py job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --task=mnist --device=cpu```
- You can switch device between 'cpu' and 'gpu'

On 44323, 44324, 44327, 44328:
5. Do point 1 in important notes.

### CIFAR10
On 44323, 44324, 44327, 44328:

1. ```cd ~/workspace/teamnet_mpi/```
2. In shake_shake.py, uncommnet the code under '# 4 devices' and comment the code under '# 2 devices'
3. Do point 2 in important notes.

On 44323:

4. Open a terminal and run: 
- ```mpirun -host master,lanhost,lanhost2,lanhost3 -bind-to none -mca pml ob1 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude l4tbr0,docker0,lo python3 -m mpi4py job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --task=cifar10 --device=cpu --by_conv=False```
- You can switch device between 'cpu' and 'gpu'

On 44323, 44324, 44327, 44328:
5. Do point 1 in important notes.
