# Courier AI

A courier agent that delivers a package across a 10x10 city grid with walls and moving traffic. It combines A* search (route planning) with Q-learning (reinforcement learning) that learns to avoid traffic.

Project for CSAI-3210 Introduction to AI, Webster University in Tashkent, Fall 2026. Status: proposal stage; the files here are an early prototype.

## How it works

- **A\* (search and planning):** finds the shortest path on the wall map and outputs the next move and the remaining distance. It ignores traffic.
- **Q-learning (reinforcement learning):** sees A*'s suggested move plus whether each of the four moves is free, blocked or next to traffic, and learns which move to take. A*'s output is part of its state and of its reward.

## Files

- `env.py` - the grid environment, the A* search and the state used by the agent.
- `exp.py` - evaluates a random agent, an A*-only baseline and a Q-learning agent on 500 unseen maps.

## Run

Python 3, no extra packages:

    python exp.py

## Prototype results (500 unseen maps)

| Agent | Delivered | Crashed |
|---|---|---|
| Random | 0% | about 50% |
| A*-only baseline | 66.0% | 34.0% |
| A* + Q-learning | 94.2% | 1.0% |

## Planned

A Streamlit app to run episodes and compare the agents, harder settings (larger grids, more traffic), and a stronger baseline.
