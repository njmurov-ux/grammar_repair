# Weight-Space Programming: Repairing a Swedish Grammar Network

This repository contains a controlled experiment investigating whether frontier language models can diagnose and repair a neural network by directly examining and modifying its weights, without gradient access or feedback on their proposed edits.

The experiment was inspired by [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) and conducted as part of the SE.LLMA research assignment.

## Experiment

We trained a small neural network (4,425 parameters) to predict Swedish adjective inflections, covering regular, irregular, geminating, and other morphological classes. The trained network correctly classified all 768 generated grammatical cases.

We then introduced a targeted defect that erased the distinction between common and neuter gender in certain contexts, producing 86 errors while preserving the remaining predictions.

GPT-6.2 Astra and GPT-6.2 Sol received the corrupted weights, architecture, and labeled diagnostic examples. Each was asked to propose a repair subject
