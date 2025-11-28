from abc import abstractmethod
from typing import Optional
import torch
from torch import Tensor
import argparse

import wandb

from exogym.strategy.communicate import broadcast, all_reduce, all_gather
from exogym.strategy.strategy import SimpleReduceStrategy, Strategy
from exogym.trainer import Trainer
from exogym.strategy.optim import OptimSpec
from exogym.aux.utils import get_device

from nanogpt import GPT, GPTConfig, get_dataset

NUM_NODES = 4

### PLAYGROUND
### This is a minimal configuration for training a nanogpt model with a given strategy.
### The strategy can be swapped out for custom logic by writing a new strategy class.


class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    @abstractmethod
    def get_indices(self, param, iteration, **kwargs): ...


class RandomIndexSelector(IndexSelector):
    def get_indices(self, param, iteration, **kwargs) -> Tensor:
        return torch.bernoulli(
            torch.full(param.shape, self.p, device=param.device)
        ).bool()
    
    def log_difference(self, param, mask, iteration):
        # no difference statistic for random selector
        pass


class MaxGradIndexSelector(IndexSelector):
    def get_indices(
        self, param: Tensor, iteration: int | None = None, **kwargs
    ) -> Tensor:
        k = max(1, int(self.p * param.numel()))
        _, indices = torch.topk(param.grad.abs().view(-1), k)
        mask = torch.zeros(param.numel(), dtype=torch.bool, device=param.device)
        mask[indices] = True
        return mask.view(param.shape)

    # add a utility that logs the difference between the average gradient of the selected indices
    # and the average gradient of the whole parameter;
    def log_difference(self, param: Tensor, mask: Tensor, iteration: int):
        avg_grad_selected = param.grad[mask].abs().mean().item()
        avg_grad = param.grad.abs().mean().item()
        if wandb.run is not None and (iteration + 1) % 100 == 0:
            wandb.log(
                {"selected_vs_all_diff": avg_grad_selected - avg_grad},
                step=iteration,
            )


# a selector that picks the largest parameters by absolute value
class MaxParamIndexSelector(IndexSelector):
    def get_indices(
        self, param: Tensor, iteration: int | None = None, **kwargs
    ) -> Tensor:
        k = max(1, int(self.p * param.numel()))
        _, indices = torch.topk(param.abs().view(-1), k)
        mask = torch.zeros(param.numel(), dtype=torch.bool, device=param.device)
        mask[indices] = True
        return mask.view(param.shape)

    def log_difference(self, param: Tensor, mask: Tensor, iteration: int):
        avg_param_selected = param.abs()[mask].mean().item()
        avg_param = param.abs().mean().item()
        if wandb.run is not None and (iteration + 1) % 100 == 0:
            wandb.log(
                {"selected_vs_all_diff": avg_param_selected - avg_param},
                step=iteration,
            )


# Another index selector that could be intersting is  maybe something that uses the momentum values;
# update the weights that are supposed to be updated the most
# check how we get momentum out of the optimiser
class MaxMomentumIndexSelector(IndexSelector):
    def _get_momentum_buffer(self, optim_state):
        momentum_buffer = optim_state.get("momentum_buffer", None)
        if momentum_buffer is None:
            # in AdamW, the momentum buffer is stored under 'exp_avg'
            momentum_buffer = optim_state.get("exp_avg", None)
        return momentum_buffer

    def get_indices(
        self, param: Tensor, iteration: int | None = None, **kwargs
    ) -> Tensor:
        # optimi_state, optim_state: dict[str, Tensor], should be passed in kwargs
        optim_state = kwargs.get("optim_state", {})

        momentum_buffer = self._get_momentum_buffer(optim_state)

        if momentum_buffer is None:
            # if no momentum buffer, fall back to random selection
            return RandomIndexSelector(self.p).get_indices(param, iteration)

        k = max(1, int(self.p * param.numel()))
        _, indices = torch.topk(momentum_buffer.abs().view(-1), k)
        mask = torch.zeros(param.numel(), dtype=torch.bool, device=param.device)
        mask[indices] = True
        return mask.view(param.shape)

    def log_difference(self, param, mask, iteration):
        momentum_buffer = self._get_momentum_buffer(param)
        if momentum_buffer is None:
            return
        avg_momentum_selected = momentum_buffer.abs()[mask].mean().item()
        avg_momentum = momentum_buffer.abs().mean().item()
        if wandb.run is not None and (iteration + 1) % 100 == 0:
            wandb.log(
                {"selected_vs_all_diff": avg_momentum_selected - avg_momentum},
                step=iteration,
            )


# define a dict with the index selectors so we can have an arg that maps
INDEX_SELECTORS = {
    "random": RandomIndexSelector,
    "max_grad": MaxGradIndexSelector,
    "max_param": MaxParamIndexSelector,
    "max_momentum": MaxMomentumIndexSelector,
}


class SPARTAStrategy(Strategy):
    def __init__(
        self,
        optim_spec: Optional[str | OptimSpec] = None,
        p_sparta=0.005,
        index_selector="max_grad",
        **kwargs,
    ):
        self.index_selector_name = index_selector
        index_selector = INDEX_SELECTORS[index_selector](p_sparta)
        super().__init__(**kwargs)

        self.optim_spec = (
            optim_spec
            if isinstance(optim_spec, OptimSpec)
            else OptimSpec.from_str(optim_spec)
        )
        self.index_selector = index_selector

    def step(self):
        with torch.no_grad():
            for param in self.model.parameters():
                # possibly also skip self.local_step == 0
                if not param.requires_grad or param.grad is None:
                    continue
                optim_state = self.optim.state.get(param, {})
                indices_mask = self.index_selector.get_indices(
                    param, self.local_step, optim_state=optim_state
                )
                # log difference statistic
                if self.index_selector_name == "max_momentum":
                    self.index_selector.log_difference(
                        optim_state, indices_mask, self.local_step
                    )
                else:
                    self.index_selector.log_difference(
                        param, indices_mask, self.local_step
                    )

                broadcast(indices_mask, src=0)  # does this send to other nodes?
                sparse_data = param.data[indices_mask]
                # all reduce across the four nodes
                all_reduce(sparse_data, op=torch.distributed.ReduceOp.SUM)
                sparse_data /= self.num_nodes
                param.masked_scatter_(indices_mask, sparse_data)

        self.optim.step()
        super().step()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="owt")
    # add gpt2_small v gpt_sbase as argument
    arg_parser.add_argument(
        "--model", type=str, default="gpt2_small"
    )  # gpt2_small or gpt2_sbase
    arg_parser.add_argument("--run_name", type=str, default="sparta-run")
    arg_parser.add_argument(
        "--index_selector", type=str, default="max_momentum"
    )  # random, max_grad, max_param, max_momentum
    args = arg_parser.parse_args()
    print(
        f"Using dataset: {args.dataset}, model: {args.model}, run name: {args.run_name}"
    )

    # Get datasets - this will take a while the first time, as the dataset has to be imported and processed.
    train_dataset, vocab_size = get_dataset(
        args.dataset,
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * NUM_NODES if args.dataset == "owt" else 0.99,
    )
    val_dataset, vocab_size = get_dataset(
        args.dataset, block_size=1024, device="cpu", start_pc=0.99, end_pc=1.0
    )

    device = get_device()

    if args.model == "gpt2_small":
        gpt_config = GPTConfig.gpt2_small()
    elif args.model == "gpt2_sbase":
        gpt_config = GPTConfig.gpt2_sbase()
    else:
        raise NotImplementedError

    gpt_config.vocab_size = vocab_size
    model = GPT(gpt_config)

    # Create trainer
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
    )

    ## STRATEGY - This is where we define custom logic

    # to default back to data parallel training:
    # strategy = SimpleReduceStrategy(
    #     optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0004),
    #     lr_scheduler="lambda_cosine",
    #     lr_scheduler_kwargs={
    #         "warmup_steps": 1000,
    #         "cosine_anneal": True,
    #     },
    #     max_norm=1.0,
    # )

    strategy = SPARTAStrategy(
        optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0005),
        lr_scheduler="lambda_cosine",
        lr_scheduler_kwargs={
            "warmup_steps": 1000,
            "cosine_anneal": True,
        },
        max_norm=1.0,
        p=0.005,
        index_selector=args.index_selector,
    )

    # Train it!
    trainer.fit(
        num_epochs=1,
        max_steps=20000,
        strategy=strategy,
        num_nodes=NUM_NODES,
        device=device,
        batch_size=16,
        minibatch_size=16,  # Gradient accumulation to ensure we can fit in memory. Make this even lower for smaller devices.
        shuffle=False,
        val_size=256,
        val_interval=100,
        wandb_project="sparta",
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
