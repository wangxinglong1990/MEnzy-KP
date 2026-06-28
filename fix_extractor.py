#!/usr/bin/env python3
"""Fix extractor.py imports and print."""
with open("src/features/extractor.py", "r") as f:
    content = f.read()

# Fix import
old_import = """from config import (
    PROTEIN_ESMC_MODEL_NAME,
    SMILES_TRANSFORMER_CHECKPOINT,
    SMILES_TRANSFORMER_DIR,
)"""
new_import = """from config import (
    PROTEIN_ESMC_MODEL_NAME,
    PROTEIN_ESMC_WEIGHTS_PATH,
    SMILES_TRANSFORMER_CHECKPOINT,
    SMILES_TRANSFORMER_DIR,
)"""
content = content.replace(old_import, new_import)

# Fix print statement
old_print = 'print(f"  ESMC loaded in {(weight_path.stat().st_mtime):.0f}s", flush=True)'
new_print = 'print(f"  ESMC loaded (device={self.device})", flush=True)'
content = content.replace(old_print, new_print)

with open("src/features/extractor.py", "w") as f:
    f.write(content)
print("✅ extractor.py fixed (imports + print)")
