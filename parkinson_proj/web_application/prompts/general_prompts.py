"""
General prompts for video action classification.
"""

# Basic action classification prompt
ACTION_CLASSIFICATION_PROMPT = """Analyze this video clip and classify the action being performed. 
Choose ONLY ONE action from this list: {actions}

Respond with just the action name, nothing else."""

# Error handling prompt
ERROR_RECOVERY_PROMPT = """The previous analysis failed. Please try again with this video clip.
Classify the action from: {actions}

Respond with just the action name."""

# Validation prompt for action classification
VALIDATION_PROMPT = """Is "{predicted_action}" a valid action from this list: {actions}?

Respond with "yes" or "no" only.""" 