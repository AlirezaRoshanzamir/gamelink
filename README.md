# Gamelink

A Python framework for implementing and simulating strategy/board games with pluggable AI agents.

## Overview

`gamelink` provides a set of abstract base classes and utilities for building turn-based games that support:

- **Reversible actions** — steps can be undone, enabling tree search over game states
- **Pluggable decision strategies** — swap between random, human CLI, and AI players at runtime
- **Brute-force game tree search** — enumerate all possible game outcomes via backtracking
- **SAT solving** — express and evaluate boolean logic over game state (used for social deduction games)

## Requirements

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (package manager)
