#!/usr/bin/env python3
"""Patch _EsmcProteinEncoder in extractor.py for fast ESMC loading."""
import sys
sys.path.insert(0, '.')

with open("src/features/extractor.py", "r") as f:
    content = f.read()

old_loading = '''        self.ESMC = ESMC
        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Keep the same loading style as Tt_prediction:
        # ESMC.from_pretrained(config.ESM_MODEL_NAME, device=config.DEVICE)
        self.client = self.ESMC.from_pretrained(PROTEIN_ESMC_MODEL_NAME, device=self.device)'''

new_loading = '''        self.ESMC = ESMC
        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Fast loading: bypass from_pretrained (which does slow to(bfloat16))
        from esm.tokenization import get_esmc_model_tokenizers
        tok = get_esmc_model_tokenizers()
        self.client = self.ESMC(
            d_model=960, n_heads=15, n_layers=30,
            tokenizer=tok, use_flash_attn=False,
        ).eval()
        weight_path = PROTEIN_ESMC_WEIGHTS_PATH
        if not weight_path.exists():
            from esm.utils.constants.esm3 import data_root
            weight_path = data_root("esmc-300") / "data/weights/esmc_300m_2024_12_v0.pth"
        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.client.load_state_dict(sd)
        if self.device.type != "cpu":
            self.client = self.client.to(self.device).to(torch.float16)
        print(f"  ESMC loaded in {(weight_path.stat().st_mtime):.0f}s", flush=True)'''

if old_loading in content:
    content = content.replace(old_loading, new_loading)
    with open("src/features/extractor.py", "w") as f:
        f.write(content)
    print("✅ extractor.py patched successfully")
else:
    print("⚠️  Could not find old loading code, checking extractor.py...")
    # Show the relevant section
    import re
    m = re.search(r'class _EsmcProteinEncoder.*?def _encode_one', content, re.DOTALL)
    if m:
        print(m.group()[:1000])
