# ROSIE Supercomputer Command Reference

This document contains all the commands and procedures for working with ROSIE, the MSOE supercomputer.

## Table of Contents
1. [Connection and Access](#connection-and-access)
2. [Basic SLURM Commands](#basic-slurm-commands)
3. [Interactive Jobs (srun)](#interactive-jobs-srun)
4. [Batch Jobs (sbatch)](#batch-jobs-sbatch)
5. [Conda Environment](#conda-environment)
6. [GPU Commands](#gpu-commands)
7. [File Management](#file-management)
8. [Node Types and Partitions](#node-types-and-partitions)
9. [Monitoring and Debugging](#monitoring-and-debugging)

---

## Connection and Access

### SSH Configuration
Configure SSH for easy access by adding to `~/.ssh/config`:

```bash
Host rosie
HostName dh-mgmt3.hpc.msoe.edu
User "your_username"

Host rosie1
HostName dh-mgmt1.hpc.msoe.edu
User "your_username"

Host rosie2
HostName dh-mgmt2.hpc.msoe.edu
User "your_username"

Host rosie3
HostName dh-mgmt3.hpc.msoe.edu
User "your_username"

Host rosie4
HostName dh-mgmt4.hpc.msoe.edu
User "your_username"
```

### SSH Key Setup
```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -a 100

# Copy key to Rosie
ssh-copy-id rosie

# Login to Rosie
ssh rosie
```

### Direct Node SSH
```bash
# SSH into a specific node (when you have a job running on it)
ssh dh-nodeXX
```

### Open On Demand
- Access via web browser
- Use "Clusters" -> "Rosie Shell Access" for terminal access
- Use "Files" section for file management

---

## Basic SLURM Commands

### Check Hostname
```bash
# From management node
hostname

# From allocated node
srun hostname

# From specific node
srun --nodelist=dh-node1 hostname
```

### Queue Management
```bash
# View all running jobs
squeue

# View your jobs only
squeue -u your_username

# Cancel a job
scancel <job_id>
```

### Job Output
```bash
# Follow job output in real-time
tail -f slurm-<job_id>.out

# View completed job output
cat slurm-<job_id>.out
```

---

## Interactive Jobs (srun)

### Basic Interactive Shell (No GPU)
```bash
# Minimum CPUs (2 CPUs, shared core)
srun --partition=teaching --pty --cpus-per-task=2 bash

# With more CPUs
srun --partition=teaching --pty --cpus-per-task=16 bash

# Basic interactive shell without partition specification
srun --pty bash
```

### Interactive Shell with GPU
```bash
# Single GPU with 8 CPUs (Teaching partition)
srun --partition=teaching --pty --gpus=1 --cpus-per-gpu=8 bash

# Single GPU with 16 CPUs (Teaching partition)
srun --gpus=1 --cpus-per-gpu=16 --pty bash

# DGX partition with GPU
srun --partition=dgx --pty --gpus=1 --cpus-per-gpu=16 bash

# H100 partition with GPU
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=16 --pty bash
```

### Running Single Commands with srun
```bash
# Run single command on teaching node
srun --partition=teaching --gpus=1 --cpus-per-gpu=16 nvidia-smi

# Run on DGX
srun --partition=dgx --gpus=1 --cpus-per-gpu=16 nvidia-smi

# Run on H100
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=16 nvidia-smi
```

### Running Python Scripts with srun
```bash
# Teaching partition with GPU
srun --gpus=1 --cpus-per-gpu=8 singularity exec --nv -B /data:/data /data/containers/msoe-tensorflow-20.07-tf2-py3.sif python /home/ad.msoe.edu/XXXXX/Lab11/Lab11.py --data /data/cs2300/L9/fruits --batch_size 4 --epochs 5 --main_dir /home/ad.msoe.edu/XXXXXX/Lab11 --augment_data false --fine_tune false

# With conda environment (Teaching)
srun --partition=teaching --gpus=1 --cpus-per-gpu=8 bash --login -c "conda activate /data/csc4611/conda-csc4611/; python script.py"

# With conda environment (DGX)
srun --partition=dgx --gpus=1 --cpus-per-gpu=8 bash --login -c "conda activate /data/csc4611/conda-csc4611/; python script.py"

# With conda environment (H100)
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=8 bash --login -c "conda activate /data/csc4611/conda-csc4611/; python script.py"
```

### One-Line Python Test Commands
```bash
# Test GPU allocation
srun --gpus=1 --pty bash --login -c "conda activate /data/csc4611/conda-csc4611/; python -c \"import torch;print('device:',torch.cuda.device(0));\""

# Check device count and current device
srun --gpus=1 --cpus-per-gpu=16 --pty bash -ic "conda activate /data/csc4611/conda-csc4611/; python -c \"import torch;print('device:',torch.cuda.current_device());print('threads:',torch.get_num_threads());\""

# DGX version
srun --partition=dgx --gpus=1 --cpus-per-gpu=16 bash --login -cc "conda activate /data/csc4611/conda-csc4611/; python -c \"import torch; print('device:',torch.cuda.current_device())\""

# H100 version
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=16 bash --login -cc "conda activate /data/csc4611/conda-csc4611/; python -c \"import torch; print('device:',torch.cuda.current_device())\""
```

### Running Shell Scripts with srun
```bash
# Make script executable
chmod u+x example.sh

# Run on DGX
srun --partition=dgx --gpus=1 --cpus-per-gpu=16 bash ./example.sh
```

---

## Batch Jobs (sbatch)

### Creating an sbatch Script
Example script structure (`script_name.sh`):

```bash
#!/bin/bash

#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16

# Your commands here
bash --login -cc "echo -n host:; hostname; conda activate /data/csc4611/conda-csc4611/; python -c \"import torch; print('device:',torch.cuda.current_device())\""
```

### Submitting Batch Jobs
```bash
# Navigate to script directory
cd Lab11

# Submit the job
sbatch lab11.sh

# Submit with custom script name
sbatch script_name.sh
```

### Converting srun to sbatch
Take an srun command like:
```bash
srun --partition=dgx --gpus=1 --cpus-per-gpu=16 bash --login -cc "conda activate /data/csc4611/conda-csc4611/; python script.py"
```

Convert to sbatch script:
```bash
#!/bin/bash

#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16

bash --login -cc "conda activate /data/csc4611/conda-csc4611/; python script.py"
```

---

## Conda Environment

### Activating the Course Environment
```bash
# Standard activation (from teaching node)
conda activate /data/csc4611/conda-csc4611/

# First-time H100 setup (run once)
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=16 --pty bash
source /etc/profile
conda activate /data/csc4611/conda-csc4611
exit
```

### Python Commands in Conda Environment
```bash
# Start Python
python

# Import PyTorch
import torch

# Check GPU availability
torch.cuda.device_count()  # Should return number of allocated GPUs

# Check current device
torch.cuda.current_device()  # Should return 0 for first GPU

# Check thread count
torch.get_num_threads()

# Exit Python
exit()
```

---

## GPU Commands

### Check GPU Status
```bash
# From allocated node
nvidia-smi

# From management node (teaching partition)
srun --partition=teaching --gpus=1 --cpus-per-gpu=16 nvidia-smi

# From management node (DGX)
srun --partition=dgx --gpus=1 --cpus-per-gpu=16 nvidia-smi

# From management node (H100)
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=16 nvidia-smi
```

### Check CPU Count
```bash
# From allocated node
nproc
```

---

## File Management

### Creating Files
```bash
# Using cat (type Ctrl-D when done)
cat > example.sh
echo "Hello, world!"
nvidia-smi
ls
# Press Ctrl-D to finish
```

### Setting Permissions
```bash
# Make file executable
chmod u+x example.sh
```

### Copying Files
```bash
# Copy file on Rosie
cp /data/datasets/pokemon/gan.py ~/path/where/Im/working/

# Download from Rosie to local machine
scp username@rosie:path/within/your/home/folder/on/rosie/file.npy .
```

### Viewing Files
```bash
# View file contents
less script.py

# View file with line numbers
cat -n script.py
```

---

## Node Types and Partitions

### Available Partitions
1. **teaching** (default): Tesla T4 GPUs
   - 4 GPUs per node
   - 72 CPUs per node
   - Recommended: 16 CPUs per GPU (4 students per node)

2. **dgx**: V100 GPUs
   - 8 GPUs per node (24 students can run simultaneously on 3 nodes)
   - Use sparingly for large jobs

3. **dgxh100**: H100 GPUs (newest, most powerful)
   - Only 2 nodes available
   - Use very sparingly
   - 8 CPUs per GPU recommended

### GPU Specifications
- **Tesla T4**: 8.1 TFLOPS (single-precision)
- **V100**: 16.4 TFLOPS (single-precision)
- **H100**: 60 TFLOPS (single-precision)

### Resource Allocation Guidelines
- **Minimum CPUs**: 2 (to avoid sharing physical core due to hyperthreading)
- **Standard CPUs per GPU**: 8
- **Maximum CPUs per GPU**: 16 (teaching partition)

---

## Monitoring and Debugging

### Monitoring Jobs
```bash
# Check if job is running
squeue

# Monitor job output in real-time
tail -f slurm-<job_id>.out

# Exit tail (does NOT cancel job)
Ctrl-C
```

### Killing Jobs
```bash
# From interactive session
Ctrl-C (twice)

# From management node
scancel <job_id>
```

### Rosie Dashboard
- URL: http://dashboard.hpc.msoe.edu/visualizer/
- Shows real-time node usage
- Note: Your "GPU 0" may be a different GPU number on the dashboard

### Debugging Tips
1. **Bus Errors**: Use `squeue` to find free nodes, specify with `--nodelist=dh-nodeXX`
2. **File Caching Issues**: Copy sbatch scripts to new names before editing
3. **Conda Not Working**: Ensure `--login` flag is used with bash
4. **Job Not Starting**: Check `squeue` for queue status

### Command History
```bash
# Use up arrow to recall previous commands
# Useful for re-running srun commands with modifications
```

### Checking Node Assignment
```bash
# While job is running, find your node
squeue

# SSH into your assigned node
ssh dh-nodeXX

# Check GPU usage on that node
nvidia-smi

# Exit back to management node
exit
```

---

## Common Command Patterns

### Pattern 1: Interactive Development
```bash
# 1. Login to Rosie
ssh rosie

# 2. Request interactive node with GPU
srun --gpus=1 --cpus-per-gpu=8 --pty bash

# 3. Activate conda environment
conda activate /data/csc4611/conda-csc4611/

# 4. Run your code
python script.py

# 5. Exit when done
exit
```

### Pattern 2: Quick Testing
```bash
# Single command from management node
srun --partition=teaching --gpus=1 --cpus-per-gpu=8 bash --login -c "conda activate /data/csc4611/conda-csc4611/; python script.py"
```

### Pattern 3: Long-Running Batch Job
```bash
# 1. Create sbatch script
cat > myjob.sh
#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
bash --login -c "conda activate /data/csc4611/conda-csc4611/; python long_script.py"
# Ctrl-D

# 2. Make executable and submit
chmod u+x myjob.sh
sbatch myjob.sh

# 3. Monitor
tail -f slurm-<job_id>.out
```

### Pattern 4: Running with Singularity Container
```bash
srun --gpus=1 --cpus-per-gpu=8 singularity exec --nv -B /data:/data \
  /data/containers/msoe-tensorflow-20.07-tf2-py3.sif \
  python /home/ad.msoe.edu/USERNAME/script.py --arg1 value1 --arg2 value2
```

---

## Important Notes

1. **Use GlobalProtect VPN when off-campus**
2. **Always activate conda environment** when using PyTorch/TensorFlow
3. **Request appropriate resources**: Don't request H100 if T4 is sufficient
4. **Use sbatch for jobs > 30 minutes** to avoid losing work if disconnected
5. **Check squeue before submitting** to see current load
6. **Save different versions of sbatch scripts** to avoid caching issues
7. **Maximum job time**: 24 hours (jobs will timeout after this)
8. **The `--login` flag is required** for conda activate to work in non-interactive contexts
9. **Use `-c` vs `-cc`**: Single `-c` for simple commands, `-cc` for more complex ones
10. **File paths must be absolute** when running from management node via srun/sbatch

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Login | `ssh rosie` |
| Interactive CPU | `srun --pty --cpus-per-task=2 bash` |
| Interactive GPU | `srun --gpus=1 --cpus-per-gpu=8 --pty bash` |
| Activate conda | `conda activate /data/csc4611/conda-csc4611/` |
| Check GPUs | `nvidia-smi` |
| Check queue | `squeue` |
| Submit batch | `sbatch script.sh` |
| Monitor output | `tail -f slurm-<id>.out` |
| Kill job | `scancel <id>` or `Ctrl-C` twice |
| SSH to node | `ssh dh-nodeXX` |

---

## Example Commands Library

### Teaching Partition Examples
```bash
# CPU only, 16 cores
srun --partition=teaching --cpus-per-task=16 --pty bash

# 1 GPU, 8 CPUs, interactive
srun --partition=teaching --gpus=1 --cpus-per-gpu=8 --pty bash

# 1 GPU, run Python script directly
srun --partition=teaching --gpus=1 --cpus-per-gpu=8 bash --login -c "conda activate /data/csc4611/conda-csc4611/; python gan.py"
```

### DGX Partition Examples
```bash
# Interactive with GPU
srun --partition=dgx --gpus=1 --cpus-per-gpu=16 --pty bash

# Run command directly
srun --partition=dgx --gpus=1 --cpus-per-gpu=16 nvidia-smi

# Run Python with conda
srun --partition=dgx --gpus=1 --cpus-per-gpu=8 bash --login -c "conda activate /data/csc4611/conda-csc4611/; python gan.py"
```

### H100 Partition Examples
```bash
# Interactive with GPU
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=16 --pty bash

# Run command directly
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=16 nvidia-smi

# Run Python with conda
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=8 bash --login -c "conda activate /data/csc4611/conda-csc4611/; python gan.py"
```

---

## Troubleshooting

### Problem: Conda activate not working
**Solution**: Use `--login` flag with bash:
```bash
bash --login -c "conda activate /data/csc4611/conda-csc4611/; your_command"
```

### Problem: nvidia-smi shows no GPU
**Solution**: Ensure you requested GPU with `--gpus=1` flag

### Problem: Job killed unexpectedly
**Solution**: Use sbatch instead of srun for long jobs, or use tmux/nohup

### Problem: Bus errors
**Solution**: Check squeue for free nodes, specify with --nodelist

### Problem: Module/package not found
**Solution**: Verify conda environment is activated

### Problem: Syntax errors on format strings
**Solution**: Conda environment not activated properly

### Problem: First time using H100 node
**Solution**: Run these commands once:
```bash
srun --partition=dgxh100 --gpus=1 --cpus-per-gpu=16 --pty bash
source /etc/profile
conda activate /data/csc4611/conda-csc4611
exit
```

---

*Last Updated: 2025-12-10*
*For MSOE ROSIE Supercomputer*
