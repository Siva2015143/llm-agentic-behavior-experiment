# LLM Agentic Behavior Experiment

**Systematic framework for measuring reasoning behavior, compute usage, and output quality in large language models.**

*Independent research by Sivamani Battala*  

---

## Table of Contents
- [Overview](#overview)
- [What It Does](#what-it-does)
- [Reasoning Modes](#reasoning-modes)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Output and Analysis](#output-and-analysis)
- [What You Learn](#what-you-learn)
- [Technical Details](#technical-details)
- [Why This Matters](#why-this-matters)
- [Troubleshooting](#troubleshooting)
- [About This Project](#about-this-project)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project explores how different prompting strategies affect large language model reasoning patterns, computational cost, and output quality.

The main goal is to measure the trade-off between:

- response speed  
- reasoning depth  
- compute usage  
- answer quality  

This repository served as the experimental foundation for the research paper:

**Agentic Compute Criticality**  
DOI: https://doi.org/10.5281/zenodo.19469219

---

## What It Does

The framework tests LLMs under three reasoning modes and records the behavioral differences.

| Mode | Approach | Use Case |
|------|----------|----------|
| **P0** | Direct answer, minimal reasoning | Fast responses, simple queries |
| **P3** | Moderate deliberation | Balanced speed and accuracy |
| **P6** | Extended reasoning with self-correction | Complex problems requiring depth |

It tracks:

- token usage  
- latency  
- reasoning markers  
- compute estimates  
- response structure  
- output quality  

This helps quantify compute-quality trade-offs in a reproducible way.

---

## Reasoning Modes

### P0
A single-pass mode focused on direct answers.

### P3
A moderate mode that allows more reflection and careful response shaping.

### P6
A deeper reasoning mode that encourages extended analysis and self-correction.

These modes allow controlled comparison of how prompting affects model behavior.

---

## Quick Start

### Installation

```bash
git clone https://github.com/Siva2015143/llm-agentic-behavior-experiment.git
cd llm-agentic-behavior-experiment
pip install -r requirements.txt
