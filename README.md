# AlphaStowage: Multi-port Stowage Planning with Deep Reinforcement Learning

An AlphaZero-inspired deep reinforcement learning approach to tackle the Multi-Port Stowage Planning Problem (MPSP) in maritime logistics.

## 📋 Overview

Maritime transport handles over 80% of global trade volume, making efficient container vessel utilization crucial for minimizing operational costs. This project implements an end-to-end solution for the **Multi-Port Stowage Planning Problem (MPSP)** using deep reinforcement learning, trained entirely through self-play without human intervention.

### Key Features

- 🧠 **AlphaZero-inspired architecture** with Monte Carlo Graph Search (MCGS)
- 🎮 **Custom MPSP environment** available on [PyPI](https://pypi.org/project/MPSPEnv)
- 🚀 **Outperforms traditional MIP models** on larger problem instances
- 📊 **Trained on 150,000+ episodes** across 168 different problem sizes
- 🔄 **No domain knowledge required** - learns optimal strategies from scratch

## 🎯 Problem Description

The MPSP involves planning the loading and unloading of containers across multiple ports along a ship's route. The goal is to **minimize reshuffles** (containers that must be moved to access others) while considering:

- Sequential port visits
- Container destination constraints
- Stack accessibility (only top containers are reachable)
- Bay capacity limitations

## 🏗️ Architecture

### State Representation
- **Bay Matrix (B)**: R × C matrix representing container positions
- **Transportation Matrix (T)**: N × N matrix of containers by origin/destination
- Normalized inputs for generalization across ship sizes

### Action Space
- **2·R·C actions**: Add or remove 1 to R containers per column
- Intelligent action masking to eliminate invalid moves and transpositions
- Automatic execution of forced moves to reduce episode length

### Neural Network
- Residual tower with 20 blocks (128 channels)
- Dual-head architecture: policy head + value head
- Incorporates state embeddings mid-tower for enhanced learning
- SiLU activation functions throughout

### Search Algorithm
- **Monte Carlo Graph Search (MCGS)** to handle transpositions efficiently
- PUCT algorithm for exploration-exploitation balance
- 600-1800 MCTS iterations per position during evaluation
- Dirichlet noise for enhanced exploration during training

## 📊 Results

Evaluated against state-of-the-art methods on benchmark datasets:

| Method | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=16 | **Average** |
|--------|-----|-----|-----|------|------|------|------|-------------|
| Avriel et al. (2000) | 0.1 | 0.6 | 1.2 | 2.5 | 6.3 | 64.1 | 269.7 | **49.2** |
| Ding & Chou (2015) | 0.1 | 0.6 | 1.2 | 3.1 | 26.8 | 116.7 | 276.5 | **60.7** |
| Parreño-Torres et al. (2019) | 0.1 | 0.7 | 1.3 | 3.1 | 5.0 | 7.6 | 11.6 | **4.2** |
| **AlphaStowage (N=600)** | 0.3 | 1.9 | 4.4 | 7.9 | 12.5 | 16.3 | 22.7 | **9.3** |
| **AlphaStowage (N=1800)** | 0.1 | 1.9 | 3.6 | 6.7 | 11.6 | 15.3 | 21.3 | **8.6** |

*Values represent average number of reshuffles. Lower is better.*

## 🚀 Installation

### Environment
```bash
pip install MPSPEnv
```

### Training Code
```bash
git clone https://github.com/hojmax/AlphaStowage.git
cd AlphaStowage
pip install -r requirements.txt
```

## 💻 Usage

### Using the Environment

```python
from MPSPEnv import MPSPEnv

# Create environment
env = MPSPEnv(num_ports=6, rows=8, cols=4)

# Reset environment
state = env.reset()

# Take action
next_state, reward, done, info = env.step(action)
```

### Training

```bash
python train.py --config configs/default.yaml
```

Key training parameters:
- **MCTS iterations**: 600 (training), 600-1800 (evaluation)
- **Episodes**: 150,000+
- **Replay buffer**: 3.6M observations (~130K episodes)
- **Temperature**: 0.2
- **Learning rate**: 0.001 (exponentially decayed)

## 📁 Project Structure

```
AlphaStowage/
├── src/
│   ├── models/          # Neural network architectures
│   ├── search/          # MCTS/MCGS implementation
│   ├── training/        # Self-play and training loops
│   └── utils/           # Helper functions
├── configs/             # Configuration files
├── data/                # Benchmark datasets
├── notebooks/           # Analysis and visualization
└── tests/               # Unit tests
```

## 🔬 Technical Highlights

### Action Space Reduction
- Increased action granularity (add/remove multiple containers)
- 8 masking rules to eliminate invalid/redundant actions
- Reduced average episode length from ~700 to ~100 moves

### Monte Carlo Graph Search
- Handles transpositions efficiently (same state, different paths)
- Recursive backpropagation for DAG structures
- Prevents information leakage while maximizing computation reuse

### Curriculum & Optimization
- Trained across 168 problem sizes simultaneously
- Min-max normalization for unbounded value estimates
- Authentic matrix generation for i.i.d. training/test distribution

## 📈 Future Work

- **Curriculum learning**: Progressive difficulty scaling
- **Additional features**: Blocking indicators, reshuffle predictions
- **Automatic action space reduction**: Learn to identify equivalent states
- **Real-world constraints**: Stability, crane limitations, special containers
- **Computational scaling**: Optimize for commercial deployment

## 📚 Citation

```bibtex
@mastersthesis{hojmark2024alphastowage,
  author = {Højmark, Axel and Jensen, Christian Mølholt},
  title = {Multi-port Stowage Planning With Deep Neural Networks and Graph Search},
  year = {2024},
  school = {University of Copenhagen}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Axel Højmark** - [axho@di.ku.dk](mailto:axho@di.ku.dk)
- **Christian Mølholt Jensen** - [chrj@di.ku.dk](mailto:chrj@di.ku.dk)

## 🙏 Acknowledgments

- Benchmark data provided by Parreño-Torres et al. (2019)
- Inspired by AlphaZero (Silver et al., 2017) and KataGo (Wu, 2020)
- Built with PyTorch and custom MCTS implementation

## 📞 Contact

For questions or collaboration opportunities, please reach out via email or open an issue on GitHub.

---

**Note**: This is a research project demonstrating the potential of deep RL for complex logistics problems. Deployment to real-world maritime operations would require substantial additional optimization and computational resources.
