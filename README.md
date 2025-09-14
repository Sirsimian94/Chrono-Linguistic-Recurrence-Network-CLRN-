# Chrono-Linguistic-Recurrence-Network-CLRN-
CLRN is a novel neural network that processes sequences on parallel timescales (fast, medium, slow) to master long-term context. It was conceived by one AI (Gemini) and built via human-directed collaboration with other AIs in just 24 hours—a new paradigm for rapid research.
The Chrono-Linguistic Recurrence Network (CLRN) is a novel multi-timescale recurrent neural architecture for sequence modeling, developed through a 24-hour intensive AI collaboration. The model integrates fast, medium, and slow streams with adaptive temporal gating, robust normalization, and cross-stream attention mechanisms to capture complex temporal dependencies effectively.

Features
Adaptive Temporal Gating Unit (TGU) with learnable mixing coefficients

Multi-head cross attention with positional encoding and masking

Dynamic cache management supporting variable batch sizes

Comprehensive edge case handling and padding mask integration

Extensive built-in test suite validating functionality and stability

Installation
Requires Python 3.8+ and PyTorch.

Install dependencies with:

text
pip install torch
Usage
Train or test the model by importing ImprovedCLRN from clrn_final.py. See the built-in ultimate_comprehensive_test_clrn function for example usage and tests.

python
from clrn_final import ImprovedCLRN

model = ImprovedCLRN(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 128))
logits, weights = model(tokens)
Contributing
Contributions and issues are welcome via GitHub.

License
This project is licensed under the MIT License.
