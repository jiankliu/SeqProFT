export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/GO-CC/GO-CC-nf-mha.yml
python train_search.py configs/GO-CC/GO-CC-f-mha.yml

python train_search.py configs/GO-CC/GO-CC-nf-smh.yml 
python train_search.py configs/GO-CC/GO-CC-f-smh.yml 

python train_search.py configs/GO-CC/GO-CC-nf-smha.yml 
python train_search.py configs/GO-CC/GO-CC-f-smha.yml 

python train_search.py configs/GO-CC/GO-CC-nf-mha-35M.yml
python train_search.py configs/GO-CC/GO-CC-f-mha-35M.yml
python train_search.py configs/GO-CC/GO-CC-nf-mha-150M.yml
python train_search.py configs/GO-CC/GO-CC-f-mha-150M.yml

python train_search.py configs/GO-CC/GO-CC-nf-smha-35M.yml
python train_search.py configs/GO-CC/GO-CC-f-smha-35M.yml
python train_search.py configs/GO-CC/GO-CC-nf-smha-150M.yml
python train_search.py configs/GO-CC/GO-CC-f-smha-150M.yml