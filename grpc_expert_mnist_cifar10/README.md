# Run TF using GRPC instructions with experts

## Important Notes
1. Execute the script 'reinstall.sh' in ~/Documents, When the GRPC job completed each time.
2. The location to put the required file scripts:
- __init__.py: working_directory/tensor2tensor/
- expert_job_dispatcher.py: working_directory/
- expert_moe.py: working_directory/
- utility.py: working_directory/
- expert_shake_shake.py: working_directory/tensor2tensor/models/
- expert_basic.py: working_directory/tensor2tensor/models/
- expert_utils.py: working_directory/tensor2tensor/utils/
3. Master device is 44323


## 2 Experts
### MNIST
On each devices:
1. ```cd ~/workspace/expert_mnist/expert_2```
2. Do part 2 in important notes.

On master:
3. Run mpi: ```mpirun -host master,lanhost -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude l4tbr0,docker0,lo python3 -m mpi4py  expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324 --num_experts=2 --task=mnist --device=cpu```, you can swtich device between 'cpu' and 'gpu'

On each devices:
4. Do part 1 in important notes.

### CIFAR10
On each devices:
1. ```cd ~/workspace/expert_2```
2. Do part 2 in important notes.

On master:
3. Run mpi: ```mpirun -host master,lanhost -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude l4tbr0,docker0,lo python3 -m mpi4py  expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324 --num_experts=2 --task=cifar10 --device=cpu```, you can swtich device between 'cpu' and 'gpu'

On each devices:
4. Do part 1 in important notes.

## 4 Experts
### MNIST
On each devices:
1. ```cd ~/workspace/expert_mnist/expert_4```
2. Do part 2 in important notes.

On master:
3. Run mpi: ```mpirun -host master,lanhost,lanhost2,lanhost3 -bind-to none -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude l4tbr0,docker0,lo python3 -m mpi4py expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --num_experts=4 --task=mnist --device=cpu```, you can swtich device between 'cpu' and 'gpu'

On each devices:
4. Do part 1 in important notes.

### CIFAR10
On each devices:
1. ```cd ~/workspace/expert_4```
2. Do part 2 in important notes.

On master:
3. Run mpi: ```mpirun -host master,lanhost,lanhost2,lanhost3 -bind-to none -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude l4tbr0,docker0,lo python3 -m mpi4py expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --num_experts=4 --task=cifar10 --device=cpu```, you can swtich device between 'cpu' and 'gpu'

On each devices:
4. Do part 1 in important notes.
