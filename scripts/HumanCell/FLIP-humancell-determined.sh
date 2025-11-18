export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/FLIP-humancell/FLIP-humancell-nf-mha.yml 
python train_search.py configs/FLIP-humancell/FLIP-humancell-f-mha.yml 

python train_search.py configs/FLIP-humancell/FLIP-humancell-nf-smh.yml 
python train_search.py configs/FLIP-humancell/FLIP-humancell-f-smh.yml 

python train_search.py configs/FLIP-humancell/FLIP-humancell-nf-smha.yml
python train_search.py configs/FLIP-humancell/FLIP-humancell-f-smha.yml

python train_search.py configs/FLIP-humancell/FLIP-humancell-nf-mha-35M.yml
python train_search.py configs/FLIP-humancell/FLIP-humancell-f-mha-35M.yml
python train_search.py configs/FLIP-humancell/FLIP-humancell-nf-mha-150M.yml
python train_search.py configs/FLIP-humancell/FLIP-humancell-f-mha-150M.yml

python train_search.py configs/FLIP-humancell/FLIP-humancell-nf-smha-35M.yml
python train_search.py configs/FLIP-humancell/FLIP-humancell-f-smha-35M.yml
python train_search.py configs/FLIP-humancell/FLIP-humancell-nf-smha-150M.yml
python train_search.py configs/FLIP-humancell/FLIP-humancell-f-smha-150M.yml
