import torch
import numpy as np
import random
from model.DeepFM import DeepFM
from model.AutoInt import AutoInt

def get_device(): 
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(config, feature_sizes, device, seq_vocab_size=None):
    model_type = config.get("model_type", "deepfm")
     
    if model_type.lower() == "deepfm":
        model = DeepFM(
            feature_sizes=feature_sizes,
            embedding_size=config["embedding_size"],
            hidden_dims=config["deepfm_hidden_dims"],
            num_classes=1,
            dropout=config["deepfm_dropout"],
            use_cuda = config.get("use_cuda", True),
            verbose = False,
            seq_vocab_size=seq_vocab_size
        )
    elif model_type.lower() == "autoint":
        model = AutoInt(
            feature_sizes=feature_sizes,
            embedding_size=config["embedding_size"],
            embedding_dropout=config["embedding_dropout"],
            att_layer_num=config["att_layer_num"],
            att_head_num=config["att_head_num"],
            att_res=config["att_res"],
            att_dropout=config["att_dropout"],
            dnn_hidden_units=config["autoint_dnn_hidden_units"],
            dnn_activation='relu',
            l2_reg_dnn=config["l2_reg_dnn"],
            l2_reg_embedding=config["l2_reg_embedding"],
            dnn_use_bn=config["dnn_use_bn"],
            dnn_dropout=config["dnn_dropout"],
            init_std=config["init_std"],
            seed=config["seed"],
            use_cuda=config["use_cuda"],
            device=device,
            seq_vocab_size=seq_vocab_size
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model.to(device)
