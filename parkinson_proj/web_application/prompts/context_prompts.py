"""
Prompts for detailed context analysis and descriptions.
"""

# General description prompt
GENERAL_DESCRIPTION_PROMPT = """Describe this video segment in general terms.

Focus on:
- Main action being performed
- Basic objects or environment
- Overall movement pattern

Segment: {segment_info}
Action: {action_type}
Duration: {duration} seconds

Provide a concise, general description."""

# Detailed description prompt with structured output
DETAILED_DESCRIPTION_PROMPT = """Analyze this video segment and provide structured output in exactly this format:

# 1. SUBJECT:
[Describe what the subject is doing, their posture, movements, gait patterns, balance, coordination. Consider medical aspects like potential Parkinson's symptoms, tremors, rigidity, bradykinesia, or postural instability.]

# 2. INTERACTION:
[Describe whether the subject is interacting with any objects using their hands or body. Include details about hand movements, grasping patterns, fine motor control, object manipulation skills, and any difficulties observed.]

# 3. CONTEXT:
[Describe the environment around the subject, including furniture, room layout, lighting, surfaces, potential hazards, and how the environment might affect the subject's mobility or safety, especially for patients with movement disorders.]

Segment: {segment_info}
Action: {action_type}
Duration: {duration} seconds

Please follow the exact format above with the numbered sections and provide medical-relevant observations for potential Parkinson's disease analysis."""

# Medical context prompt
MEDICAL_CONTEXT_PROMPT = """Analyze this video segment from a medical perspective.

Consider:
- Movement quality and fluidity
- Potential mobility issues
- Balance and coordination
- Fatigue indicators
- Safety concerns
- Functional assessment
- Rehabilitation implications

Segment: {segment_info}
Action: {action_type}
Duration: {duration} seconds

Provide medical-relevant observations and recommendations."""

# Object interaction prompt
OBJECT_INTERACTION_PROMPT = """Describe the object interactions in this video segment.

Focus on:
- Objects being handled or used
- How objects are manipulated
- Spatial relationship to objects
- Safety of object interactions
- Functional use of objects

Segment: {segment_info}
Action: {action_type}

Describe the object interactions in detail."""

# Movement quality prompt
MOVEMENT_QUALITY_PROMPT = """Assess the quality of movement in this video segment.

Consider:
- Smoothness of movement
- Coordination and balance
- Speed and rhythm
- Posture and alignment
- Energy efficiency
- Potential compensatory movements

Segment: {segment_info}
Action: {action_type}

Provide a detailed movement quality assessment.""" 