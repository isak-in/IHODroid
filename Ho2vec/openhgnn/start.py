from .utils import set_random_seed, set_best_config, Logger
from .trainerflow import build_flow
from .auto import hpo_experiment


def OpenHGNN(args):
    if not getattr(args, 'seed', False):
        args.seed = 0
    args.logger = Logger(args)
    if getattr(args, "use_best_config", False):
        args = set_best_config(args)
    set_random_seed(args.seed)
    trainerflow = SpecificTrainerflow.get(args.model, args.task)
    if getattr(args, "use_hpo", False):
        # hyper-parameter search
        hpo_experiment(args, trainerflow)
    else:
        flow = build_flow(args, trainerflow)
        result = flow.train()
        return result


SpecificTrainerflow = {

}
