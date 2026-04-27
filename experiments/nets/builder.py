import importlib
from pprint import pprint
from peft import LoraConfig, get_peft_model
from experiments.nets.base import UNet
from torch import nn
import re


def dl_module(net: str, expr: bool = True):
    pck, net = net.rsplit(".", 1)
    if expr:
        pck = f"experiments.{pck}"
    return getattr(importlib.import_module(pck), net)


class Builder(object):
    """Builder is intended to replace the call by unet instance to call by dict so we can save unet as hyperparameters"""

    def __init__(self):
        self.actions = []

    def based_on(self, module: str, *args, **kwargs):
        self.actions.append(
            {
                "action": "base",
                "module": module,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return self

    def based_on_plan(self, module: str, *args, **kwargs):
        self.actions.append(
            {
                "action": "plan",
                "module": module,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return self

    def based_on_ckpt(self, ckpt_path: str, *args, **kwargs):
        self.actions.append(
            {
                "action": "base_ckpt",
                "ckpt": ckpt_path,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return self

    def reinitialize(self, module: str, *args, **kwargs):
        self.actions.append(
            {
                "action": "reinitialize",
                "module": module,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return self

    def lora(
        self,
        target_module_type: list[str],
        *args,
        skips: list[str] | None = None,
        **kwargs,
    ):
        self.actions.append(
            {
                "action": "lora",
                "target": target_module_type,
                "skips": skips if skips is not None else [],
                "args": args,
                "kwargs": kwargs,
            }
        )
        return self

    def load_lm(self, ckpt_path: str, *args, **kwargs):
        self.actions.append(
            {
                "action": "load_lm",
                "ckpt": ckpt_path,
                "args": args,
                "kwargs": kwargs,
            }
        )
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

        match base["action"]:
            case "base":
                return dl_module(base["module"])(*base["args"], **base["kwargs"])
            case "plan":
                return dl_module(base["module"]).from_plan(
                    *base["args"], **base["kwargs"]
                )
            case "base_ckpt":
                return dl_pretrained_unet(base["ckpt"])
            case _:
                raise NotImplementedError(
                    f"action {base['action']} is not implemented for base"
                )

    def _act(self, action, unet: UNet):
        from experiments.trainer import dl_pretrained_unet

        match action["action"]:
            case "reinitialize":
                unet = dl_module(action["module"]).reinitialize_unet(
                    unet, *action["args"], **action["kwargs"]
                )
            case "load_lm":
                dl = dl_pretrained_unet(action["ckpt"])
                unet.load_state_dict(dl.state_dict(), strict=True)
            case "lora":

                def match(
                    name: str,
                    module: nn.Module,
                    matches: list[tuple[str, tuple[type, ...]]],
                ) -> bool:
                    for n, tys in matches:
                        if re.fullmatch(n, name) and type(module) in tys:
                            return True
                    return False

                def parse_matches(matches):
                    return [
                        (splits[0], tuple(dl_module(ty, False) for ty in splits[1:]))
                        for splits in [t.split("/") for t in matches]
                    ]

                def _findall(
                    unet: nn.Module, matches: list[tuple[str, tuple[type, ...]]]
                ):
                    return list(
                        filter(
                            lambda t: match(t[0], t[1], matches), unet.named_modules()
                        )
                    )

                def findall(unet: nn.Module, matches: list[str]):
                    return _findall(
                        unet,
                        parse_matches(matches)
                    )

                def findall_with_check(unet, matches):
                    ret = findall(unet, matches)
                    if len(ret) == 0:
                        print("of all following unet modules: ")
                        for name, mod in unet.named_modules():
                            print(name, type(mod))
                        print("no one matches the expression", parse_matches(matches))
                        raise AssertionError("No one module was found")
                    return [t[0] for t in ret]

                target_modules = findall_with_check(unet, action["target"])
                print("lora target modules: ")
                pprint(target_modules)
                if "modules_to_save" in action["kwargs"]:
                    action["kwargs"]["modules_to_save"] = findall_with_check(
                        unet, action["kwargs"]["modules_to_save"]
                    )
                    print("lora modules to save: ")
                    pprint(action["kwargs"]["modules_to_save"])

                conf = LoraConfig(
                    target_modules=target_modules, *action["args"], **action["kwargs"]
                )

                unet = get_peft_model(unet, conf)
            case _:
                raise NotImplementedError(
                    f"action {action['action']} is not implemented"
                )
        return unet

    def build(self) -> UNet:
        base, *actions = self.actions
        unet = self._build_base(base)
        for action in actions:
            unet = self._act(action, unet)
        return unet
