import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureHook:
    def __init__(
        self,
        module,
    ):
        self.feature = None

        self.handle = (
            module.register_forward_hook(
                self._hook
            )
        )

    def _extract_tensor(
        self,
        output,
    ):
        if torch.is_tensor(output):
            return output

        if isinstance(
            output,
            (list, tuple)
        ):
            for item in output:

                result = (
                    self._extract_tensor(
                        item
                    )
                )

                if result is not None:
                    return result

        if isinstance(
            output,
            dict
        ):
            for item in output.values():

                result = (
                    self._extract_tensor(
                        item
                    )
                )

                if result is not None:
                    return result

        return None

    def _hook(
        self,
        module,
        inputs,
        output,
    ):
        self.feature = (
            self._extract_tensor(
                output
            )
        )

    def clear(self):
        self.feature = None

    def remove(self):
        self.handle.remove()