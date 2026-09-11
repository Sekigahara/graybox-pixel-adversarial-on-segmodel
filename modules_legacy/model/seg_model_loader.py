from modules.model.segmentation_model.deeplabv3 import DeepLabV3Model
from modules.model.segmentation_model.lraspp import LRASPPModel
from modules.model.segmentation_model.segformer import SegFormerModel
from modules.model.segmentation_model.mask2former_vistas import Mask2FormerVistasModel

def load_segmentation_model(
    name: str,
):
    name = name.lower()

    if name == "deeplabv3":

        model = DeepLabV3Model()

    elif name == "mask2former_vistas":
        return Mask2FormerVistasModel()
        
    elif name == "lraspp":

        model = LRASPPModel()

    elif name == "segformer_b0_ade":

        model = SegFormerModel(
            "nvidia/"
            "segformer-b0-finetuned-ade-512-512"
        )

    elif name == "segformer_b2_ade":

        model = SegFormerModel(
            "nvidia/"
            "segformer-b2-finetuned-ade-512-512"
        )

    elif name == "segformer_b0_cityscapes":

        model = SegFormerModel(
            "nvidia/"
            "segformer-b0-finetuned-cityscapes-1024-1024"
        )

    elif name == "segformer_b2_cityscapes":

        model = SegFormerModel(
            "nvidia/"
            "segformer-b2-finetuned-cityscapes-1024-1024"
        )

    else:

        raise ValueError(
            f"Unknown segmentation model: {name}"
        )

    model.freeze()

    return model
