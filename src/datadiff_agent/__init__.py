"""datadiff-agent: an AI agent that guards data pipelines.

Each new batch of data is compared with the last good batch using
datadiff-engine. The agent then reasons about what changed, using the
dataset's contract and its memory of past runs, and decides whether the
batch should be loaded (PASS), loaded with an alert (WARN), or kept out of
the warehouse (BLOCK).
"""

__version__ = "0.1.0"
