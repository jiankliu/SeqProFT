export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/FoldClassification/FoldClassification-nf-mha.yml 
python train_search.py configs/FoldClassification/FoldClassification-f-mha.yml 

python train_search.py configs/FoldClassification/FoldClassification-nf-smh.yml 
python train_search.py configs/FoldClassification/FoldClassification-f-smh.yml 

python train_search.py configs/FoldClassification/FoldClassification-nf-smha.yml 
python train_search.py configs/FoldClassification/FoldClassification-f-smha.yml

python train_search.py configs/FoldClassification/FoldClassification-nf-mha-35M.yml
python train_search.py configs/FoldClassification/FoldClassification-f-mha-35M.yml 
python train_search.py configs/FoldClassification/FoldClassification-nf-mha-150M.yml
python train_search.py configs/FoldClassification/FoldClassification-f-mha-150M.yml 

python train_search.py configs/FoldClassification/FoldClassification-nf-smha-35M.yml
python train_search.py configs/FoldClassification/FoldClassification-f-smha-35M.yml 
python train_search.py configs/FoldClassification/FoldClassification-nf-smha-150M.yml
python train_search.py configs/FoldClassification/FoldClassification-f-smha-150M.yml 
