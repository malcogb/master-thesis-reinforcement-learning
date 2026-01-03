# Centralized Training with Decentralized Execution for Multi-Agent Reinforcement Learning in Crisis Management Scenarios

**Master Thesis Project – Reinforcement Learning**

**Author:** Malco GBAKPA  
**Degree:** Master in Computer Science (Artificial Intelligence)  
**Institution:** Dakar Institute of Technology  
**Academic Year:** 2024–2025

---

## Project Overview

This repository contains the code and experimental material for an individual Master's thesis investigating the use of deep reinforcement learning for training multiple autonomous aerial agents in simulated three-dimensional environments.

The primary objective is to study decentralized decision-making under uncertainty in safety-critical and crisis management–inspired scenarios. The work focuses on learning stability, scalability, and emergent coordination rather than explicit communication or centralized control at execution time.

All experiments are conducted in simulation for reproducibility and controlled analysis.

---

## Learning Paradigm

The project follows a **Centralized Training with Decentralized Execution (CTDE)** paradigm with shared policy parameters across agents.

### Key characteristics of the learning setup:

- A single **shared policy** is trained using parameter sharing across all agents
- Agents act independently at execution time based on **global observations** provided by the environment
- Training benefits from:
  - A global observation space
  - Shared reward signals
- **No explicit inter-agent communication** or coordination protocol is implemented
- **No joint action-value function** or game-theoretic optimization is used

At execution time, no centralized controller selects joint actions. Each agent independently applies the shared policy to its own observation vector, which contains global state information exposed by the environment.

This setup allows implicit coordination to emerge through interaction with the environment rather than through designed communication mechanisms.

**Important note:** While multiple agents are present, this work does not implement a full multi-agent reinforcement learning framework with explicit coordination or communication. The current implementation uses **total observation** (all agents receive access to the global state). Partial observation and communication learning are planned for future work.

---

## Environment and Agents

- **Agents:** 3 autonomous aerial agents (drones)
- **Environment:** Simulated 3D environment (Unity ML-Agents / OpenAI Gym compatible)
- **Tasks:**
  - Navigation
  - Area coverage
  - Safety-aware behavior under uncertainty
- **Training Strategy:** Curriculum learning with progressive task complexity

The environmental constraints are gradually increased during training to study learning dynamics under increasing decision complexity.

---

## Learning Algorithm

- **Algorithm:** Proximal Policy Optimization (PPO)
- **Policy:** Shared neural network policy
- **Execution:** Fully decentralized
- **Evaluation Focus:**
  - Reward convergence
  - Training stability
  - Variance across random seeds
  - Robustness of learned behaviors

PPO was selected due to its stability and suitability for continuous control tasks in simulated robotic environments.

---

## Experimental Results

The experimental results presented in the thesis are based on the trained model `ppo_marl_20251211_161150` (trained on December 11, 2025). This model can be evaluated using the evaluation scripts provided in this repository.

---

## Prerequisites

- Unity 2021.3.45f2 or compatible version
- Python 3.8+
- PyTorch
- Stable-Baselines3
- TensorBoard

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/master-thesis-reinforcement-learning.git
   cd master-thesis-reinforcement-learning
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Unity project (root directory) in Unity Editor

4. Configure parameters in `python/config.py` if needed

---

## Usage

### Training

```bash
cd marl
python train_marl.py
```

**Options:**
- `--continue_training MODEL_PATH`: Continue training from a checkpoint
- `--load_model MODEL_PATH`: Load a model and optionally reset curriculum
- `--reset_curriculum`: Reset curriculum when loading a model

### Evaluation

**Single evaluation:**
```bash
cd marl
python evaluate_marl.py --model models/ppo_marl_20251211_161150.zip --episodes 100
```

**Multiple runs evaluation:**
```bash
cd marl
python evaluate_marl.py --model models/ppo_marl_20251211_161150.zip --episodes 100 --runs 5
```

**List available models:**
```bash
python evaluate_marl.py --list
```

### Launching Unity Environment

Use the provided PowerShell scripts in the `scripts/` directory:

```powershell
# Launch Unity builds for parallel training
.\scripts\launch_unity_builds.ps1

# Check Unity builds status
.\scripts\check_unity_builds.ps1

# Stop Unity builds
.\scripts\stop_unity_builds.ps1
```

---

## Project Structure

```
AeroPatrol_drone/
├── python/              # Python code (Gymnasium wrapper, configuration, environment management)
│   ├── aero_patrol_wrapper.py
│   ├── config.py
│   ├── env_manager.py
│   └── helpers.py
├── marl/                # Training and evaluation scripts
│   ├── train_marl.py
│   └── evaluate_marl.py
├── powerShell/             # PowerShell utility scripts
│   ├── launch_unity_builds.ps1
│   ├── check_unity_builds.ps1
│   └── ...
├── Assets/              # Unity C# code (simulation)
│   ├── Scripts/         # Main game logic
│   ├── Scenes/          # Unity scenes
│   ├── Materials/       # Unity materials used for environment rendering and visualization
│   ├── Plugins/         # Unity plugins and external dependencies required for simulation and Python communication
│   ├── Prefabs/         # Game objects prefabs
│   └── PeacefulPie/     # Unity-Python communication
├── ProjectSettings/     # Unity project configuration
├── Packages/            # Unity package dependencies
├── README.md
└── requirements.txt
```

---

## Key Contributions

This work provides:

- An empirical study of shared-policy learning in a multi-agent setting
- Analysis of implicit coordination emerging without communication
- A curriculum learning strategy applied to decentralized execution
- Experimental evaluation of CTDE scalability in simulated robotic systems
- Application-oriented discussion in the context of safety and crisis management scenarios

---

## Limitations

This project intentionally does not implement:

- Explicit inter-agent communication mechanisms
- Game-theoretic or cooperative MARL algorithms
- Formal safety constraints or planning guarantees
- Sim-to-real transfer

The focus remains on learning dynamics and empirical analysis rather than deployment or formal guarantees.

**Current implementation status:**
- ✅ Total observation (global state)
- ⏳ Partial observation (planned for future work)

---

## Academic Context

This project was conducted as part of a Master's degree in Artificial Intelligence and serves as a foundation for future doctoral research in:

- Reinforcement Learning and Markov Decision Processes
- Multi-agent decision-making
- Autonomous systems
- Safe and trustworthy AI

---

## Perspectives and Future Work

Potential research extensions include:

- Explicit multi-agent coordination and communication learning
- Integration of planning and decision-making
- Risk-aware and safety-constrained reinforcement learning
- Sim-to-real transfer for robotic systems
- Partial observation implementation
- Limited communication protocols

These directions align with current challenges in autonomous and multi-agent systems research.

---

## Citation

If you use this code in your research, please cite:

```
GBAKPA, M. (2025). Centralized Training with Decentralized Execution for 
Multi-Agent Reinforcement Learning in Crisis Management Scenarios. 
Master's Thesis, Dakar Institute of Technology.
```

---

## License

This project is released under the MIT License for academic and research purposes.

---

## Contact

Malco GBAKPA  
Email: malcogbakpa@gmail.com
