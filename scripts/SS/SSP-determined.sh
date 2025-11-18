export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/SSP/SSP-nf-mha.yml 
python train_search.py configs/SSP/SSP-f-mha.yml 

python train_search.py configs/SSP/SSP-nf-smh.yml 
python train_search.py configs/SSP/SSP-f-smh.yml 

python train_search.py configs/SSP/SSP-nf-smha.yml 
python train_search.py configs/SSP/SSP-f-smha.yml

python train_search.py configs/SSP/SSP-nf-smha-35M.yml
python train_search.py configs/SSP/SSP-f-smha-35M.yml 
python train_search.py configs/SSP/SSP-nf-smha-150M.yml
python train_search.py configs/SSP/SSP-f-smha-150M.yml 

