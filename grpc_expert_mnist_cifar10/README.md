# Run TF using GRPC instructions with experts

## Important Notes
1. Execute the script 'reinstall.sh' in ~/Documents, When the GRPC job completed each time.
2. The location to put the required scripts:
- \_\_init\_\_.py: working_directory/tensor2tensor/
- expert_job_dispatcher.py: working_directory/
- expert_moe.py: working_directory/
- utility.py: working_directory/
- expert_shake_shake.py: working_directory/tensor2tensor/models/
- expert_basic.py: working_directory/tensor2tensor/models/
- expert_utils.py: working_directory/tensor2tensor/utils/
3. Master device is 44323


## 2 Experts
### MNIST
On 44323 and 44324:

1. ```cd ~/workspace/expert_mnist/expert_2```
2. Do point 2 in important notes.
3. Open a terminal on each device and run: 
- ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324 --use_mpi=False  --task=mnist --num_experts=2 --device=cpu --task_index=0``` on 44323
- ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324 --use_mpi=False  --task=mnist --num_experts=2 --device=cpu --task_index=1``` on 44324
- You can switch device between 'cpu' and 'gpu'
4. Do point 1 in important notes.

### CIFAR10
On 44323 and 44324:

1. ```cd ~/workspace/expert_2```
2. Do point 2 in important notes.
3. Open a terminal on each device and run: 
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324 --use_mpi=False  --task=cifar10 --num_experts=2 --device=cpu --task_index=0``` on 44323
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324 --use_mpi=False  --task=cifar10 --num_experts=2 --device=cpu --task_index=1``` on 44324
 - You can switch device between 'cpu' and 'gpu'
4. Do point 1 in important notes.

## 4 Experts
### MNIST
On 44323, 44324, 44327, 44328:

1. ```cd ~/workspace/expert_mnist/expert_4```
2. Do point 2 in important notes.
3. Open a terminal on each device and run: 
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --use_mpi=False  --task=mnist --num_experts=4 --device=cpu --task_index=0``` on 44323
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --use_mpi=False  --task=mnist --num_experts=4 --device=cpu --task_index=1``` on 44324
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --use_mpi=False  --task=mnist --num_experts=4 --device=cpu --task_index=2``` on 44327
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --use_mpi=False  --task=mnist --num_experts=4 --device=cpu --task_index=3``` on 44328
 - You can switch device between 'cpu' and 'gpu'
4. Do point 1 in important notes.

### CIFAR10
On 44323, 44324, 44327, 44328:

1. ```cd ~/workspace/expert_4```
2. Do point 2 in important notes.
3. Open a terminal on each device and run: 
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --use_mpi=False  --task=cifar10 --num_experts=4 --device=cpu --task_index=0``` on 44323
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --use_mpi=False  --task=cifar10 --num_experts=4 --device=cpu --task_index=1``` on 44324
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --use_mpi=False  --task=cifar10 --num_experts=4 --device=cpu --task_index=2``` on 44327
 - ```python3 expert_job_dispatcher.py --worker_hosts=master:44323,lanhost:44324,lanhost2:44327,lanhost3:44328 --use_mpi=False  --task=cifar10 --num_experts=4 --device=cpu --task_index=3``` on 44328
 - You can switch device between 'cpu' and 'gpu'
4. Do point 1 in important notes.
