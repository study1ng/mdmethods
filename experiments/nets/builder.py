import importlib

from experiments.nets.base import UNet

def dl_module(net: str):
    pck, net = net.rsplit(".", 1)
    pck = f"experiments.{pck}"
    return getattr(importlib.import_module(pck), net)

class Builder(object):
    """Builder is intended to replace the call by unet instance to call by dict so we can save unet as hyperparameters"""
    def __init__(self):
        self.actions = []

    def based_on(self, module: str, *args, **kwargs):
        self.actions.append({
            "action": "base",
            "module": module,
            "args": args,
            "kwargs": kwargs,
        })
        return self
    
    def based_on_plan(self, module: str, *args, **kwargs):
        self.actions.append({
            "action": "plan",
            "module": module,
            "args": args,
            "kwargs": kwargs,
        })
        return self
    
    def based_on_ckpt(self, ckpt_path: str, *args, **kwargs):
        self.actions.append({
            "action": "base_ckpt",
            "ckpt": ckpt_path,
            "args": args,
            "kwargs": kwargs,
        })
        return self

    def reinitialize(self, module: str, *args, **kwargs):
        self.actions.append({
            "action": "reinitialize",
            "module": module,
            "args": args,
            "kwargs": kwargs,
        })
        return self

    def load_lm(self, ckpt_path: str, *args, **kwargs):
        self.actions.append({
            "action": "load_lm",
            "ckpt": ckpt_path,
            "args": args,
            "kwargs": kwargs,
        })
        return self

    def to_params(self) -> list[dict]:
        return self.actions
    
    @classmethod
    def from_params(cls, actions) -> "Builder":
        assert isinstance(actions, list)
        ret = cls()
        ret.actions = actions
        return ret
    
    def _build_base(self, base):
        from experiments.trainer import dl_pretrained_unet
        assert base["action"] in ["base", "plan"], "Builder got no base or plan"
        match base["action"]:
            case "base":
                return dl_module(base["module"])(*base["args"], **base["kwargs"])
            case "plan":
                return dl_module(base["module"]).from_plan(*base["args"], **base["kwargs"])
            case "base_ckpt":
                return dl_pretrained_unet(base["ckpt"])
            case _:
                raise NotImplementedError(f"action {base['action']} is not implemented for base")
            
    def _act(self, action, unet):
        from experiments.trainer import dl_pretrained_unet
        match action["action"]:
            case "reinitialize":
                unet = dl_module(action["module"]).reinitialize_unet(unet, *action["args"], **action["kwargs"])
            case "load_lm":
                dl = dl_pretrained_unet(action["ckpt"])
                unet.load_state_dict(dl.state_dict(), strict=True)
            case _:
                raise NotImplementedError(f"action {action['action']} is not implemented")
        return unet

    def build(self) -> UNet:
        base, *actions = self.actions
        unet = self._build_base(base)
        for action in actions:
            unet = self._act(action, unet)
        return unet