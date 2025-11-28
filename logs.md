## setup


[SSH KEY]


cd /workspace
mkdir code
cd code 
git clone git@github.com:desi-ivanova/gym.git

cd gym
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install hf_transfer


## substitute the port and the ip
ssh -p 14448 -i ~/.ssh/id_ed25519 root@213.181.105.194 "mkdir -p /workspace/code/gym/example/data"

scp -r -P 14448 -i ~/.ssh/id_ed25519 ~/Downloads/data/. root@213.181.105.194:/workspace/code/gym/example/data/


```
wandb login
cd example
python3 playground_sparta.py --run_name=random_baseline --dataset=owt --index_selector=random
```

```
python3 playground_sparta.py --run_name=max_grad --dataset=owt --index_selector=max_grad

python3 playground_sparta.py --run_name=max_param --dataset=owt --index_selector=max_param

python3 playground_sparta.py --run_name=max_momentum --dataset=owt --index_selector=max_momentum
```




## OWT hyperparams
--strategy sparta --num_nodes 4 --dataset owt --model_size sbase --batch_size 16 --minibatch_size 16 --lr 0.0005 --wandb_project DilocovsAdam --p_sparta 0.01


### momentum 

`self.optim.state[param]`