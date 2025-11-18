export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache

python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraQ.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraK.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraV.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraQK.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraQV.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraKV.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraQKV.yml

python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraQD.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraKD.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraVD.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraQKD.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraQVD.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraKVD.yml
python train_ablation.py configs/GO-BP/GO-BP-f-mha-35M-loraQKVD.yml