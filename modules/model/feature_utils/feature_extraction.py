import torch

class FeatureHook:
    def __init__(self, module):
        self.feature = None
        self.handle = module.register_forward_hook(self._hook)

    def _extract_tensor(self, output):
        if torch.is_tensor(output):
            return output

        if isinstance(output, (list, tuple)):
            for item in output:
                feature = self._extract_tensor(item)

                if feature is not None:
                    return feature

        if isinstance(output, dict):
            for item in output.values():
                feature = self._extract_tensor(item)

                if feature is not None:
                    return feature

        return None

    def _hook(self, module, inputs, output):
        self.feature = self._extract_tensor(output)

    def clear(self):
        self.feature = None

    def remove(self):
        self.handle.remove()
