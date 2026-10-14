"""
member_config.py - THE ONLY FILE A TEAM MEMBER EDITS.

Fill in: names, class list, how to load YOUR model, and how to list YOUR dataset images.
Keep everything else untouched so results from all members stay comparable.

Required:   MODEL_NAME, DATASET_NAME, CLASS_NAMES, load_model(), get_dataset()
Optional:   DATASET_VERSION, CRITICAL_CLASSES, SEED, N_SAMPLES, EXTERNAL_DIR, RUN_ID, load_image(path)

load_model() must return an object with:
    predict_proba(list_of_PIL_images) -> numpy array (len(images), num_classes) of softmax probabilities
Use mcb.adapters.TorchClassifier for any torch model that outputs logits (see the example below).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- identity
MODEL_NAME = "MyModel"            # e.g. "ConvNeXt-Base", "ResNet18", "MobileNetV3-Small"
DATASET_NAME = "MyDataset"        # e.g. "ISIC2019", "GTSRB", "OxfordPets"
DATASET_VERSION = "v1"
CLASS_NAMES = ["class0", "class1", "class2"]          # index order MUST match the model's output order
CRITICAL_CLASSES = []             # optional: classes whose errors matter most, e.g. ["MEL", "BCC"]
SEED = 42
N_SAMPLES = 1500                  # stratified subset size; 0 = use every image
EXTERNAL_DIR = ROOT / "external"  # external/images/*.jpg + external/labels.csv (filename,label)


# ----------------------------------------------------------------------------- model
def load_model():
    """Return an object with predict_proba(list_of_PIL) -> (n, C) probabilities."""
    # ---- EXAMPLE (torch model that returns logits) -- replace with your own model ----------
    # import torch
    # from torchvision import transforms
    # from mcb.adapters import TorchClassifier
    # model = build_my_model()                               # your nn.Module
    # model.load_state_dict(torch.load("path/to/checkpoint.pth", map_location="cpu"))
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224)), transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   # SAME as your validation preprocessing
    # return TorchClassifier(model, preprocess, batch_size=32, amp=True)
    #
    # ---- EXAMPLE (any other framework) ---------------------------------------------------
    # from mcb.adapters import CallableClassifier
    # return CallableClassifier(lambda images: my_predict_function(images))
    raise NotImplementedError("Edit member_config.py: implement load_model()")


# ----------------------------------------------------------------------------- data
def get_dataset() -> pd.DataFrame:
    """Return a DataFrame with columns:
         image_path : path to the image file
         label      : integer class index (0..C-1) - the GROUND TRUTH
       optional:
         group      : e.g. patient / lesion id - if present, calibration and evaluation splits never share a group
    Use only images the model did NOT train on (your validation/test images)."""
    # ---- EXAMPLE: a CSV that already has image paths and integer labels ---------------------
    # df = pd.read_csv("data/val_split.csv")
    # return pd.DataFrame({"image_path": ["data/images/" + n + ".jpg" for n in df["image"]], "label": df["label"]})
    raise NotImplementedError("Edit member_config.py: implement get_dataset()")
