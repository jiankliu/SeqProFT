export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache

python train_ablation.py configs/EC/EC-f-mha-35M-loraR1.yml
python train_ablation.py configs/EC/EC-f-mha-35M-loraR2.yml
python train_ablation.py configs/EC/EC-f-mha-35M-loraR4.yml
python train_ablation.py configs/EC/EC-f-mha-35M-loraR8.yml
python train_ablation.py configs/EC/EC-f-mha-35M-loraR16.yml


