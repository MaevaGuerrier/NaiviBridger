import torch
import yaml
import os
from utils import load_model
import onnxruntime as ort
import onnx
import einops
import torch.nn as nn
import numpy as np

class NoisePredNetWrapper(nn.Module):
    def __init__(self, nomad_model):
        super().__init__()
        self.noise_pred_net = nomad_model.noise_pred_net

    def forward(self, sample, timestep, global_cond):
        return self.noise_pred_net(
            sample=sample, timestep=timestep, global_cond=global_cond
        )


class VisionEncoderWrapper(nn.Module):
    def __init__(self, nomad_model):
        super().__init__()
        self.vision_encoder = nomad_model.vision_encoder

    def forward(self, obs_img, goal_img, input_goal_mask):
        # Call the underlying noise_pred_net directly with named arguments
        return self.vision_encoder(
            obs_img=obs_img, goal_img=goal_img, input_goal_mask=input_goal_mask
        )


class DistPredWrapper(nn.Module):
    def __init__(self, nomad_model):
        super().__init__()
        self.dist_pred_net = nomad_model.dist_pred_net

    def forward(self, obsgoal_cond):
        return self.dist_pred_net(obsgoal_cond)
    


MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

model_name = "navibridger_cvae"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open(MODEL_CONFIG_PATH, "r") as f:
    model_paths = yaml.safe_load(f)

model_config_path = model_paths[model_name]["config_path"]
with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)

if model_params["model_type"] == "cvae":
    if "train_params" in model_params:
        model_params.update(model_params["train_params"])
    if "diffuse_params" in model_params:
        model_params.update(model_params["diffuse_params"])

if model_params.get("prior_policy", None) == "cvae":
    if "diffuse_params" in model_params:
        model_params.update(model_params["diffuse_params"])

context_size = model_params["context_size"]
assert context_size != None

# load model weights
ckpth_path = model_paths[model_name]["ckpt_path"]
if os.path.exists(ckpth_path):
    print(f"Loading model from {ckpth_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
model = load_model(
    ckpth_path,
    model_params,
    device,
)
model = model.to(device)
model.eval()


print("loading model")
import pdb; pdb.set_trace()

# print("---------------------- Start Vision Encoder ----------------------------------")

# dummy_goal = torch.randn(4, 3, 96, 96, device=device)
# # Nomad vision encoder takes in 4 past obs, each with 3 channels, img dim 96x96
# # Not consistent with paper https://arxiv.org/pdf/2310.07896, past 5 times obs Figure.2
# dummy_obs = torch.randn(4, 12, 96, 96, device=device) 
# # Issue in paper codebase 'goal_mask' referenced before assignment
# # Always set in code because input_goal mask is passed see line 80 and 114 in vint_train/models/nomad/nomad_vint.py
# dummy_mask = torch.zeros(1).long().to(device)  
# dummy_mask = dummy_mask.repeat(len(dummy_goal))

# vision_wrapper = VisionEncoderWrapper(model)
# vision_wrapper = vision_wrapper.to(device)
# vision_wrapper.eval()
# # model.eval()

# output_path = "/workspace/src/NaiviBridger/deployment/model_weights/navibridger_vision_encoder.onnx"

# print("Testing forward pass for vision_encoder ...")
# with torch.no_grad():
#     test_vision_output = model(
#         "vision_encoder",
#         obs_img=dummy_obs,
#         goal_img=dummy_goal,
#         input_goal_mask=dummy_mask,
#     )

#     wrapper_vision_output = vision_wrapper(
#         obs_img=dummy_obs,
#         goal_img=dummy_goal,
#         input_goal_mask=dummy_mask,
#     )

#     print(
#         f"Success forward pass for vision encoder with shapes for model {test_vision_output.shape} and {wrapper_vision_output.shape}"
#     )


# print("\nExporting to vision encoder ONNX...")
# torch.onnx.export(
#     vision_wrapper,
#     (dummy_obs, dummy_goal, dummy_mask),
#     output_path,
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["obs_img", "goal_img", "input_goal_mask"], # This has to be the same as forward inputs (e.g., forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None))
#     output_names=["obs_encoding_tokens"], # This has to be the same as forward outputs (e.g., return output)
#     dynamic_axes={
#     "obs_img": {0: "batch_size"}, # we need it because of radius change at first 4 obs (start) then 6 obs (rad 2) 
#     "goal_img": {0: "batch_size"},
#     "input_goal_mask": {0: "batch_size"},
#     "obs_encoding_tokens": {0: "batch_size"}, 
#     },
# )


# onnx_model = onnx.load(output_path)
# onnx.checker.check_model(onnx_model)
# print("ONNX model of vision encoder is valid!")

# # Optional: Test with ONNX Runtime


# print("\nTesting vision encoder ONNX Runtime...")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ort_session = ort.InferenceSession(output_path, providers=providers)
# ort_inputs = {
#     "obs_img": dummy_obs.cpu().numpy(),
#     "goal_img": dummy_goal.cpu().numpy(),
#     "input_goal_mask": dummy_mask.cpu().numpy(),
# }
# ort_outputs = ort_session.run(None, ort_inputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")


# # Verify outputs match
# print(f"\nVerifying outputs match...")
# wrapper_cpu_output = wrapper_vision_output.cpu().numpy()
# test_vision_cpu_output = test_vision_output.cpu().numpy()
# max_diff_wrapper = abs(wrapper_cpu_output - ort_outputs[0]).max()
# max_diff_model = abs(test_vision_cpu_output - ort_outputs[0]).max()
# print(
#     f"Maximum difference between PyTorch and ONNX: {max_diff_wrapper} & {max_diff_model}"
# )


# print("---------------------- End of Vision Encoder ----------------------------------")


# print("------------------------ Distance Pred Network --------------------------------")
# print("converting dist pred network to onnx")
# # obsgoal_cond = model('vision_encoder', ...
# # dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond) --> dist takes inputs of obsgoal_cond
# # test_obs_encoding_tokens.shape torch.Size([4, 256])
# # VERY IMPORTANT the first input can be changed (see --radius in navigate.py)
# dummy_goalcond = torch.randn(4, 256, device=device)
# dist_wrapper = DistPredWrapper(model)
# dist_wrapper = dist_wrapper.to(device)
# dist_wrapper.eval()

# output_path = "/workspace/src/NaiviBridger/deployment/model_weights/navibridger_dist_pred_net.onnx"

# print("Testing forward pass for dist pred encoder ...")
# with torch.no_grad():
#     test_distance_output = model(
#         "dist_pred_net",
#         obsgoal_cond=dummy_goalcond,
#     )

#     wrapper_dist_output = dist_wrapper(
#         dummy_goalcond,
#     )

#     print(
#         f"Success forward pass for distance encoder with shapes for model {test_distance_output.shape} and {wrapper_dist_output.shape}"
#     )

# print("\nExporting to dist encoder ONNX...")
# torch.onnx.export(
#     dist_wrapper,
#     dummy_goalcond,
#     output_path,
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["obsgoal_cond"],
#     output_names=["distances_pred"],
#     dynamic_axes={
#     "obsgoal_cond": {0: "batch_size"}, 
#     },
# )


# onnx_model = onnx.load(output_path)
# onnx.checker.check_model(onnx_model)
# print("ONNX model of dist encoder is valid!")


# print("\nTesting dist encoder ONNX Runtime...")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ort_session = ort.InferenceSession(output_path, providers=providers)
# ort_inputs = {
#     "obsgoal_cond": dummy_goalcond.cpu().numpy(),
# }
# ort_outputs = ort_session.run(None, ort_inputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")

# # Verify outputs match
# print(f"\nVerifying outputs match...")
# wrapper_cpu_output = wrapper_dist_output.cpu().numpy()
# test_cpu_output = test_distance_output.cpu().numpy()
# max_diff_wrapper = abs(wrapper_cpu_output - ort_outputs[0]).max()
# max_diff_model = abs(test_cpu_output - ort_outputs[0]).max()
# print(
#     f"Maximum difference between PyTorch and ONNX: {max_diff_wrapper} & {max_diff_model}"
# )


# print("---------------------- End of Dist Encoder ----------------------------------")

# print("------------------------------- noise pred net --------------------------- ")
# print(signature(noise_pred.forward))
# IMPORTANT: local_cond is never used in nomad codebase
# (sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], local_cond=None, global_cond=None, **kwargs)


# assert model_params["prior_policy"] == "cvae", "Only CVAE model supported for ONNX export"

# batch_size = 8
# sequence_length = 8
# input_dim = 2
# encoding_size = 256

# dummy_input = torch.randn(batch_size, sequence_length, input_dim).to(device)
# dummy_global_cond = torch.randn(batch_size, encoding_size).to(device)
# dummy_timestep = torch.tensor(0, dtype=torch.int64).to(device)
# dummy_prior_cond = dummy_input.repeat_interleave(batch_size, dim=0).to(device)

# # Test forward pass first to make sure it works
# print("Testing forward pass...")
# with torch.no_grad():
#     test_output = model(
#         "noise_pred_net",
#         sample=dummy_input,
#         timestep=dummy_timestep,
#         global_cond=dummy_global_cond,
#     )
#     print(f"Forward pass successful! Output shape: {test_output.shape}")


# # Export to ONNX
# output_path = "/workspace/src/visualnav-transformer/deployment/model_weights/navibridger_noise_pred_net.onnx"


# # Export to ONNX
# wrapper = NoisePredNetWrapper(model)
# wrapper = wrapper.to(device)
# wrapper.eval()

# print("\nExporting to ONNX...")
# torch.onnx.export(
#     wrapper,
#     (dummy_input, dummy_timestep, dummy_global_cond),
#     output_path,
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["sample", "timestep", "global_cond"],
#     output_names=["noise_pred"],
#     dynamic_axes={
#         "sample": {0: "batch_size"},          
#         "global_cond": {0: "batch_size"}, 
#         "noise_pred": {0: "batch_size"}, 
#     },
# )

# print("converting noise_pred_net to onnx")


# onnx_model = onnx.load(output_path)
# onnx.checker.check_model(onnx_model)
# print("ONNX model is valid!")

# # # Optional: Test with ONNX Runtime


# # print("\nTesting ONNX Runtime...")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ort_session = ort.InferenceSession(output_path, providers=providers)
# ort_inputs = {
#     "sample": dummy_input.cpu().numpy(),
#     "timestep": dummy_timestep.cpu().numpy(),
#     "global_cond": dummy_global_cond.cpu().numpy(),
# }
# ort_outputs = ort_session.run(None, ort_inputs)
# print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")


# # Verify outputs match
# print(f"\nVerifying outputs match...")
# torch_output = test_output.cpu().numpy()
# max_diff = abs(torch_output - ort_outputs[0]).max()
# print(f"Maximum difference between PyTorch and ONNX: {max_diff}")


print("------------------------------- noise pred net --------------------------- ")




assert model_params["prior_policy"] == "cvae", "Only CVAE model supported for ONNX export"

batch_size = 8
sequence_length = 8
input_dim = 2
encoding_size = 256



# dummy_input = torch.randn(batch_size, sequence_length, input_dim).to(device)
# dummy_global_cond = torch.randn(batch_size, encoding_size).to(device)
# dummy_timestep = torch.tensor(0, dtype=torch.int64).to(device)
# dummy_prior_cond = dummy_input.repeat_interleave(batch_size, dim=0).to(device)




dummy_input = torch.randn(batch_size, sequence_length, input_dim).to(device)
dummy_sigma = torch.tensor([1.0]).to(device)
dummy_global_cond = torch.randn(batch_size, encoding_size).to(device)  # Adjust shape

# Wrap the denoising step
class DenoisingWrapper(torch.nn.Module):
    def __init__(self, diffusion, model):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
    
    def forward(self, x_t, sigma, global_cond):
        _, denoised = self.diffusion.denoise(self.model, x_t, sigma, global_cond)
        return denoised.clamp(-1, 1)

wrapper = DenoisingWrapper(diffusion, model)
wrapper.eval()

# Export to ONNX
torch.onnx.export(
    wrapper,
    (dummy_input, dummy_sigma, dummy_global_cond),
    "denoising_model.onnx",
    input_names=["x_t", "sigma", "global_cond"],
    output_names=["denoised"],
    dynamic_axes={
        "x_t": {0: "batch_size"},
        "global_cond": {0: "batch_size"},
        "denoised": {0: "batch_size"}
    },
    opset_version=17
)