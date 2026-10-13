"""Hyperparameter variant of the markov loop (TRAIN_STAT_PLAN.md section 7s).

The LLM proposes (lr, momentum) each iteration with the architecture held fixed,
instead of writing a Net class. Nothing under ``ab/gpt/act/markov/`` is modified;
the training code is imported from it and reused unchanged.
"""
