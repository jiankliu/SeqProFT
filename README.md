```sh
conda env create --file=environment.yml
```

Example training:
```sh
bash scripts/EC/EC-determined.sh
```

Example evaluating:
```sh
python eval.py configs/EC/EC-f-mha.yml path-to-checkpoint
```
