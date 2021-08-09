## Install

Navigate to the root of project, and perform:

    pip install -r requirements.txt
    pip install -e .

## Run DICE Algorithms
First, create datasets using the policy trained above:

```
for alpha in {0.0,1.0}; do python scripts/create_dataset.py --save_dir=./tests/testdata/cartpole --load_dir=./tests/testdata/cartPole --env_name=cartpole --num_trajectory=400 --max_trajectory_length=250 --alpha=$alpha --tabular_obs=0; done
```

RunFedDICE with  tabular policy parameterization on grid environments:

    python scripts/run_fed_neural_dice.py --save_dir=./tests/testdata/grid5 --load_dir=./tests/testdata --env_name=grid --nu_learning_rate=0.00003 --zeta_learning_rate=0.00003 --policy_learning_rate=0.00003 --primal_regularize=1.0 --dual_regularizer=1e-6 --tabular_obs=1

Run FedDICE with neural policy parameterization on Cartpole:

    python scripts/run_fed_pgdice.py --save_dir=./tests/testdata/cartpole --load_dir=./tests/testdata --env_name=cartpole --nu_learning_rate=0.00003 --zeta_learning_rate=0.00003 --policy_learning_rate=0.00003 --primal_regularize=1.0 --dual_regularizer=1e-6 --tabular_obs=0

