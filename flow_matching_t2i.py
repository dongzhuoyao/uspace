from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint
from absl import logging


class CNF(nn.Module):
    def __init__(
        self,
        net,
    ):
        super().__init__()
        self.net = net

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        context: Tensor,
        **kwargs,
    ) -> Tensor:
        if t.numel() == 1:
            if self.is_dissection_mode(kwargs):
                logging.info(f"debug mode, forward timesteps: {t.item()}")
            t = t.expand(x.size(0))
        _pred, inters = self.net(x, t, context=context, **kwargs)
        return _pred

    def get_ode_kwargs(self, **kwargs):
        _RTOL = 1e-5
        _ATOL = 1e-5

        if self.is_dissection_mode(kwargs):
            _solver_kwargs = kwargs["solver_kwargs"]
            logging.info("euler sampling mode, only for testing/analysis")
            if _solver_kwargs["solver"] == "fixed":
                ode_kwargs = dict(
                    method=_solver_kwargs["solver_fix"],
                    rtol=_RTOL,
                    atol=_ATOL,
                    adjoint_params=(),
                    options=dict(step_size=_solver_kwargs["solver_fix_step"]),
                )
            elif _solver_kwargs["solver"] == "adaptive":
                ode_kwargs = dict(
                    method=_solver_kwargs["solver_adaptive"],
                    rtol=_RTOL,
                    atol=_ATOL,
                    adjoint_params=(),
                    # options=dict(step_size=0.01),
                )
            elif _solver_kwargs["solver"] == "fixadp":
                _fix_kw = dict(
                    method=_solver_kwargs["solver_fix"],
                    rtol=_RTOL,
                    atol=_ATOL,
                    adjoint_params=(),
                    options=dict(step_size=_solver_kwargs["solver_fix_step"]),
                )
                _adp_kw = dict(
                    method=_solver_kwargs["solver_adaptive"],
                    rtol=_RTOL,
                    atol=_ATOL,
                    adjoint_params=(),
                    # options=dict(step_size=0.01),
                )
                return _fix_kw, _adp_kw

            else:
                raise NotImplementedError(f"solver={kwargs['solver']}")

        else:
            ode_kwargs = dict(
                method="dopri5",
                rtol=_RTOL,
                atol=_ATOL,
                adjoint_params=(),
            )
        return ode_kwargs

    def training_losses(self, x, context, sigma_min, **kwargs):
        noise = torch.randn_like(x)

        t = torch.rand(len(x), device=x.device, dtype=x.dtype)
        t_ = t[:, None, None, None]  # [B, 1, 1, 1]
        x_new = t_ * x + (1 - (1 - sigma_min) * t_) * noise
        u = x - (1 - sigma_min) * noise

        return (
            (
                self.forward(t, x_new, context=context, **kwargs) - u
            )  # self.forward = vector_field
            .square()
            .mean(dim=(1, 2, 3))
        )

    def is_dissection_mode(self, kwargs):
        return "dissect_name" in kwargs and kwargs["dissect_name"] is not None

    def encode(self, x: Tensor, context: Tensor, **kwargs) -> Tensor:
        # if y is not None:
        kwargs.update({"fm_direction": "encode"})
        func = lambda t, x: self(t, x, context=context, **kwargs)

        ode_kwargs = self.get_ode_kwargs(**kwargs)
        logging.warning("current encoding to z, should not be used in training")
        if ode_kwargs["method"] != "dopri5":
            logging.info(f"encoding to z, debug mode, {ode_kwargs}")

        return odeint(
            func,
            x,
            # 0.0,
            torch.tensor([1.0, 0.0], device=x.device, dtype=x.dtype),
            # phi=self.parameters(),
            **ode_kwargs,
        )[-1]

    def decode(
        self,
        z: Tensor,
        context: Tensor,
        **kwargs,
    ) -> Tensor:
        kwargs.update({"fm_direction": "decode"})
        func = lambda t, x: self(t, x, context=context, **kwargs)

        if kwargs["solver_kwargs"]["solver"] in ["fixed", "adaptive"]:
            ode_kwargs = self.get_ode_kwargs(**kwargs)
            return odeint(
                func,
                z,
                # 0.0,
                torch.tensor([0.0, 1.0], device=z.device, dtype=z.dtype),
                # phi=self.parameters(),
                **ode_kwargs,
            )[-1]
        elif kwargs["solver_kwargs"]["solver"] == "fixadp":
            return self.decode_fixadp(z, context, t_mid=kwargs["t_edit"], **kwargs)
        else:
            raise NotImplementedError(f"unknown solver {kwargs['solver_kwargs']}")

    def decode_fixadp(
        self,
        z: Tensor,
        context: Tensor,
        t_mid,
        **kwargs,
    ) -> Tensor:
        assert t_mid >= 0 and t_mid <= 1
        func = lambda t, x: self(t, x, context=context, **kwargs)
        ode_fixed_kwargs, ode_adaptive_kwargs = self.get_ode_kwargs(**kwargs)
        _intermediate = odeint(
            func,
            z,
            # 0.0,
            torch.tensor([0.0, t_mid], device=z.device, dtype=z.dtype),
            # phi=self.parameters(),
            **ode_fixed_kwargs,
        )[-1]

        _result = odeint(
            func,
            _intermediate,
            # 0.0,
            torch.tensor([t_mid, 1.0], device=z.device, dtype=z.dtype),
            # phi=self.parameters(),
            **ode_adaptive_kwargs,
        )[-1]
        return _result
