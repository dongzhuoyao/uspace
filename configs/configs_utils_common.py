


def construct_solver_desc(**solver_kwargs):
    if solver_kwargs["solver"] == "fixed":
        return f"{solver_kwargs['solver_fix']}_step{solver_kwargs['solver_fix_step']}"
    elif solver_kwargs["solver"] == "adaptive":
        return f"{solver_kwargs['solver_adaptive']}"
    elif solver_kwargs["solver"] == "fixadp":
        return f"{solver_kwargs['solver_fix']}_step{solver_kwargs['solver_fix_step']}-{solver_kwargs['solver_adaptive']}"
    else:
        raise NotImplementedError(f"unknown solver {solver_kwargs['solver']}")
