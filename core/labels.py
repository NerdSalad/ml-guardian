# core/labels.py
"""
Label mappings for FEVER dataset
"""

# FEVER dataset labels
LABEL_MAP = {
    0: "SUPPORTS",
    1: "REFUTES", 
    2: "NOT_ENOUGH_INFO"
}

# Alternative label mappings for different formats
LABEL_MAP_ALT = {
    0: "SUPPORTS",
    1: "REFUTES", 
    2: "NOT ENOUGH INFO"
}

# Reverse mapping for convenience
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
# Add alternative mappings
REVERSE_LABEL_MAP.update({v: k for k, v in LABEL_MAP_ALT.items()})

# Label descriptions
LABEL_DESCRIPTIONS = {
    "SUPPORTS": "The claim is supported by the evidence",
    "REFUTES": "The claim is refuted by the evidence", 
    "NOT_ENOUGH_INFO": "There is not enough information to determine if the claim is supported or refuted"
}

