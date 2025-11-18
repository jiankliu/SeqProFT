export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/Loc/Loc-nf-mha.yml 
python train_search.py configs/Loc/Loc-f-mha.yml 

python train_search.py configs/Loc/Loc-nf-smh.yml 
python train_search.py configs/Loc/Loc-f-smh.yml 

python train_search.py configs/Loc/Loc-nf-smha.yml 
python train_search.py configs/Loc/Loc-f-smha.yml

python train_search.py configs/Loc/Loc-nf-mha-35M.yml
python train_search.py configs/Loc/Loc-f-mha-35M.yml 
python train_search.py configs/Loc/Loc-nf-mha-150M.yml
python train_search.py configs/Loc/Loc-f-mha-150M.yml 

python train_search.py configs/Loc/Loc-nf-smha-35M.yml
python train_search.py configs/Loc/Loc-f-smha-35M.yml 
python train_search.py configs/Loc/Loc-nf-smha-150M.yml
python train_search.py configs/Loc/Loc-f-smha-150M.yml 
