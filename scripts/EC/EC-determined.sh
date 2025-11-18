export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/EC/EC-nf-mha.yml 
python train_search.py configs/EC/EC-f-mha.yml 

python train_search.py configs/EC/EC-nf-smh.yml 
python train_search.py configs/EC/EC-f-smh.yml 

python train_search.py configs/EC/EC-nf-smha.yml 
python train_search.py configs/EC/EC-f-smha.yml 

python train_search.py configs/EC/EC-f-mha-35M.yml
python train_search.py configs/EC/EC-f-mha-150M.yml
python train_search.py configs/EC/EC-nf-mha-35M.yml
python train_search.py configs/EC/EC-nf-mha-150M.yml

python train_search.py configs/EC/EC-f-smha-35M.yml
python train_search.py configs/EC/EC-f-smha-150M.yml
python train_search.py configs/EC/EC-nf-smha-35M.yml
python train_search.py configs/EC/EC-nf-smha-150M.yml

