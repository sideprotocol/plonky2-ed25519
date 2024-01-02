# Welcome to the `plonky2-ed25519` Repository!

`plonky2-ed25519` is a specialized implementation within the Plonky2 family, tailored specifically for the ed25519 cryptographic signature scheme. This repository builds upon the core principles of [`plonky2-gpu`](https://github.com/sideprotocol/plonky2-gpu), leveraging GPU acceleration through the CUDA framework to enhance the efficiency of cryptographic operations. 

**Instructions:**
1. **Prerequisites:**
   - Ensure you have a CPU with a minimum of 8 cores.
   - Allocate at least 16GB of RAM.
   - Utilize an NVIDIA GPU, preferably the 2080 Ti, with a minimum of 12GB GPU RAM.

2. **Building and Running:**
   - Clone the repository: `git clone https://github.com/sideprotocol/plonky2-ed25519.git`
   - Navigate to the project directory: `cd plonky2-ed25519`
   - Build the project: `./test.sh`

**Test Results:**
Our tests demonstrate a remarkable reduction in proving time for ed25519 signatures — from 45 seconds to 5 seconds with plonky2-ed25519. This optimized performance showcases the power of GPU parallelization in cryptographic operations.

## Contributor
 - [SIDE Labs](https://sidelabs.co)
