# SeqProFT
This is the source code of "SeqProFT: Sequence-only Protein Property Prediction with LoRA Finetuning" (https://doi.org/10.1109/TAI.2025.3636109).

## Environment
```sh
conda env create --file=environment.yml
```

## Training
Example training:
```sh
bash scripts/EC/EC-determined.sh
```

## Evaluating
Example evaluating:
```sh
python eval.py configs/EC/EC-f-mha.yml path-to-checkpoint
```
