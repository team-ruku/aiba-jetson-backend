import torch

from app.utils import get_accel_device


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device(get_accel_device()))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
