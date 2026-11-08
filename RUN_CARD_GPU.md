# Card-data verification on a GPU (a†b)

Re-runs Study A's verification on the public credit-card fraud data (8 units: seeds 1–3 on the fixed split,
folds 0–4) on a CUDA GPU, with the order parameter Δ = Re⟨a†b⟩ (`HCS_ORDER=adagb`, budget range [0.3686, 0.6428]).
Same code as the CPU run; only `--device cuda` and the output folder differ. Results go to
`verification_second_dataset/results_gpu/`, so the CPU results in `results/` are not touched.

## Windows (PowerShell), from this folder

```powershell
pip install -r requirements.txt
# install the CUDA build of PyTorch for your driver from pytorch.org, then check:
python -c "import torch; print(torch.cuda.is_available())"      # must print True
powershell -ExecutionPolicy Bypass -File .\RUN_CARD_GPU.ps1 -CardCsv C:\path\to\creditcard.csv -Smoke   # 5-min test
powershell -ExecutionPolicy Bypass -File .\RUN_CARD_GPU.ps1 -CardCsv C:\path\to\creditcard.csv          # full run
```

## Linux / Colab / lab machine

```bash
pip install -r requirements.txt
bash run_card_gpu.sh /path/to/creditcard.csv --smoke     # test
bash run_card_gpu.sh /path/to/creditcard.csv             # full run
```

About 25 minutes per unit on a modern GPU (the first unit also trains the baselines), about 3.5 hours in all.
A stopped run resumes with the same command. At the end the script writes `results_card_gpu.zip`; send that file.
