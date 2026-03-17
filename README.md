#  Universal Adaptive Neural Network (UANN)

A neural network inspired by the Von Neumann architecture that separates "processor" from "program", dynamically prunes useless neurons, and autonomously estimates task complexity.

This project explores energy-efficient neural networks through **meta-learning**, **dynamic pruning**, and **spontaneous specialization**. Instead of learning specific tasks (e.g., XOR or AND), the network learns a *meta-rule*—how to interpret and execute any logical operation based on its compressed description (a truth table).

##  Key Innovations

1. **The Von Neumann Analogy (Universal Logic Interpreter):** 
   The network receives binary inputs alongside a truth table (the "program"). It achieves **100% zero-shot generalization** on completely unseen Boolean functions by learning how to apply the truth table to the inputs, rather than memorizing fixed labels.
2. **Spontaneous Specialization (Thought Channels):** 
   Without explicit labels, the network divides its hidden layers into multiple parallel "channels". Channels naturally group similar logical functions (e.g., dense vs. sparse truth tables) through architectural pressure.
3. **Dynamic Resource Allocation (The Oracle):** 
   An integrated *Oracle* network analyzes the truth table *before* execution to predict linear separability. Simple tasks receive minimal compute (a single neuron), while mathematically complex tasks (like XOR) trigger the full network.
4. **Aggressive Pruning & Revival:** 
   Neurons that do not contribute are permanently killed using an Importance Score combined with L1 regularization. In the final test, the network **reduced 81% of its neurons** (from 64 down to 12) while maintaining 100% accuracy. A built-in "Revival" mechanism monitors the moving average of the loss and resurrects dead channels if catastrophic forgetting occurs.
5. **ALU Simulation (Chaining & Composition):** 
   The network can execute two different operations on the same inputs simultaneously (acting as a Half-Adder) and chain outputs without intermediate binarization.

##  Final Test Results (Block 6)
- **Training Accuracy:** 100% (on 10 base Boolean functions)
- **Test Accuracy (Zero-shot generalization):** 100% (on 6 completely unseen functions)
- **Architecture Optimization:** Starting with 64 neurons, only 12 essential neurons survived. 52 neurons were successfully pruned with zero performance degradation.

##  Development Log
This project went through multiple iterations, failures, and architectural redesigns (including experiments with Elastic Weight Consolidation and Q-Learning routing agents). To read about the engineering decisions and the evolution of the concept, check out the [Development Log (DEV_LOG.md)](DEV_LOG.md).

##  Installation and Usage

Clone the repository and install the minimal requirements (numpy, matplotlib):

```bash
git clone https://github.com/yourusername/universal-adaptive-network.git
cd universal-adaptive-network
pip install -r requirements.txt
