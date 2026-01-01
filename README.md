# Centralized Training with Decentralized Execution for Multi-Agent Reinforcement Learning in Crisis Management Scenarios

## Master Thesis Project – Reinforcement Learning

**Author:** Malco GBAKPA  
**Degree:** Master in Computer Science (Artificial Intelligence)  
**Institution:** Dakar Institute of Technology  
**Academic Year:** 2024–2025  

---

## Project Overview

This repository contains the code and experimental material for an **individual Master’s thesis** investigating the use of **deep reinforcement learning** for training multiple autonomous aerial agents in simulated three-dimensional environments.

The primary objective is to study **decentralized decision-making under uncertainty** in safety-critical and crisis management–inspired scenarios. The work focuses on learning stability, scalability, and emergent coordination rather than explicit communication or centralized control at execution time.

All experiments are conducted in simulation for reproducibility and controlled analysis.

---

## Learning Paradigm

The project follows a **Centralized Training with Decentralized Execution (CTDE)** paradigm with **shared policy parameters** across agents.

Key characteristics of the learning setup:

- A **single shared policy** is trained using parameter sharing across all agents  
- Agents act **independently at execution time** based solely on their local observations  
- Training benefits from:
  - a global observation space
  - shared reward signals
- No explicit inter-agent communication or coordination protocol is implemented  
- No joint action-value function or game-theoretic optimization is used  

This setup allows **implicit coordination** to emerge through interaction with the environment rather than through designed communication mechanisms.

Importantly, while multiple agents are present, this work does **not** implement a full multi-agent reinforcement learning framework with explicit coordination or communication.

---

## Environment and Agents

- **Agents:** 3 autonomous aerial agents (drones)  
- **Environment:** Simulated 3D environment (Unity ML-Agents / OpenAI Gym compatible)  
- **Tasks:**
  - navigation
  - area coverage
  - safety-aware behavior under uncertainty  
- **Training Strategy:** Curriculum learning with progressive task complexity  

The number of active agents and environmental constraints are gradually increased during training to study learning dynamics under increasing decision complexity.

---

## Learning Algorithm

- **Algorithm:** Proximal Policy Optimization (PPO)  
- **Policy:** Shared neural network policy  
- **Execution:** Fully decentralized  
- **Evaluation Focus:**
  - reward convergence
  - training stability
  - variance across random seeds
  - robustness of learned behaviors  

PPO was selected due to its stability and suitability for continuous control tasks in simulated robotic environments.

---

## Key Contributions

This work provides:

- An empirical study of **shared-policy learning** in a multi-agent setting  
- Analysis of **implicit coordination** emerging without communication  
- A curriculum learning strategy applied to decentralized execution  
- Experimental evaluation of CTDE scalability in simulated robotic systems  
- Application-oriented discussion in the context of safety and crisis management scenarios  

---

## Limitations

This project intentionally does **not** implement:

- Explicit inter-agent communication mechanisms  
- Game-theoretic or cooperative MARL algorithms  
- Formal safety constraints or planning guarantees  
- Sim-to-real transfer  

The focus remains on **learning dynamics and empirical analysis** rather than deployment or formal guarantees.

---

## Academic Context

This project was conducted as part of a **Master’s degree in Artificial Intelligence** and serves as a foundation for future doctoral research in:

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

These directions align with current challenges in autonomous and multi-agent systems research.



