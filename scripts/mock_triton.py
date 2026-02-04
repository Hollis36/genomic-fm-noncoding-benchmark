"""
Mock triton module for Mac compatibility.
DNABERT-2 requires triton, but it's only needed for CUDA flash attention.
This mock allows the model to load on Mac (MPS).
"""

import sys
from types import ModuleType

# Create a mock triton module
class MockTriton(ModuleType):
    """Mock triton module that does nothing."""

    def __getattr__(self, name):
        # Return a mock for any attribute access
        return lambda *args, **kwargs: None

# Install the mock
sys.modules['triton'] = MockTriton('triton')
sys.modules['triton.language'] = MockTriton('triton.language')

print("✅ Mock triton module installed for Mac compatibility")
