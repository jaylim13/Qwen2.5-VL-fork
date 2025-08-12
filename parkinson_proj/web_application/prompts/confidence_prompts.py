"""
Prompts for confidence analysis and temporal consistency checking.
"""

# Temporal consistency analysis prompt
TEMPORAL_ANALYSIS_PROMPT = """Analyze this sequence of video clips and identify potentially misclassified actions.

Sequence: {clip_sequence}
Actions: {action_sequence}

Consider:
1. Does each action make sense given the surrounding context?
2. Are there any sudden, unexplained transitions?
3. Could any clips be misclassifications?

Respond with specific clip indices and reasoning for any suspicious predictions.
Format: "Clip X: [reasoning]" for each suspicious clip."""

# Strict confidence analysis prompt
STRICT_CONFIDENCE_PROMPT = """Only flag clips that are HIGHLY likely to be wrong based on:

1. Complete context mismatch (e.g., sitting → walking → sitting in 2 seconds)
2. Statistical outliers (actions that appear only once in a long sequence)
3. Impossible transitions (e.g., standing → walking without movement context)

Sequence: {clip_sequence}
Actions: {action_sequence}

Only respond with clip indices that are almost certainly wrong.
Format: "Clip X: [strong reasoning]" for each highly suspicious clip."""

# Medium confidence analysis prompt
MEDIUM_CONFIDENCE_PROMPT = """Identify clips that are likely misclassified based on:

1. Temporal inconsistency with neighbors
2. Unusual transitions in context
3. Statistical patterns in the sequence

Sequence: {clip_sequence}
Actions: {action_sequence}

Respond with clip indices and reasoning for suspicious predictions.
Format: "Clip X: [reasoning]" for each suspicious clip."""

# Loose confidence analysis prompt
LOOSE_CONFIDENCE_PROMPT = """Identify any clips that might be misclassified.

Consider any transition between different actions as potentially suspicious.

Sequence: {clip_sequence}
Actions: {action_sequence}

Respond with clip indices and reasoning for any transitions.
Format: "Clip X: [reasoning]" for each transition."""

# Confidence level mapping
CONFIDENCE_PROMPTS = {
    1: LOOSE_CONFIDENCE_PROMPT,
    2: MEDIUM_CONFIDENCE_PROMPT,
    3: STRICT_CONFIDENCE_PROMPT
} 