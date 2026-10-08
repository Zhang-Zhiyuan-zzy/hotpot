"""Shared defaults for coordination-bond inference."""

DEFAULT_FIRST_CBOND_THRESHOLD = -0.5
DEFAULT_SUBSEQUENT_CBOND_THRESHOLD = -0.125
# Historical public name retained for callers that import the constant.
DEFAULT_CBOND_THRESHOLD = DEFAULT_SUBSEQUENT_CBOND_THRESHOLD
DEFAULT_MAX_STATES = 4096
