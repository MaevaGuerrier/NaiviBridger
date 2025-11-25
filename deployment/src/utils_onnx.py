# ROS
from sensor_msgs.msg import Image

# pytorch
# import torch
# import torch.nn as nn
# from torchvision import transforms
# import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List
import onnxruntime as ort

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Dict, List, Optional

# from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


IMAGE_ASPECT_RATIO = 4 / 3


ACTION_STATS = {"min": [-2.5, -4], "max": [5, 4]}


import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from typing import Dict, List


class TRTInfer:
    def __init__(
        self,
        engine_path: str,
        logger_severity: trt.ILogger.Severity = trt.Logger.WARNING,
    ):
        """
        Initialize TensorRT inference engine.

        Args:
            engine_path: Path to serialized TensorRT engine file
            logger_severity: TensorRT logger severity level
        """
        self.TRT_LOGGER = trt.Logger(logger_severity)

        # Load engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Store tensor info without pre-allocating memory
        self.input_specs = []
        self.output_specs = []

        num_io_tensors = self.engine.num_io_tensors

        for i in range(num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)

            tensor_spec = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
            }

            if mode == trt.TensorIOMode.INPUT:
                self.input_specs.append(tensor_spec)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_specs.append(tensor_spec)

        # Allocate buffers on first inference
        self.input_buffers = {}
        self.output_buffers = {}

    def _allocate_buffer(
        self, name: str, shape: tuple, dtype: np.dtype
    ) -> cuda.DeviceAllocation:
        """Allocate or reallocate device buffer if needed."""
        size = int(np.prod(shape) * dtype().itemsize)  # Convert to Python int

        # Check if buffer exists and is large enough
        buffer_dict = (
            self.input_buffers
            if any(s["name"] == name for s in self.input_specs)
            else self.output_buffers
        )

        if name in buffer_dict:
            old_buffer, old_size = buffer_dict[name]
            if old_size >= size:
                return old_buffer
            else:
                # Free old buffer and allocate new one
                old_buffer.free()

        # Allocate new buffer
        new_buffer = cuda.mem_alloc(size)
        buffer_dict[name] = (new_buffer, size)
        return new_buffer

    def infer(self, **kwargs) -> List[np.ndarray]:
        """
        Run inference.

        Args:
            **kwargs: input_name -> np.ndarray mappings

        Returns:
            List of np.ndarray corresponding to outputs
        """
        # Validate inputs
        input_names = {spec["name"] for spec in self.input_specs}
        provided_names = set(kwargs.keys())

        if input_names != provided_names:
            raise ValueError(
                f"Input mismatch. Expected: {input_names}, Got: {provided_names}"
            )

        # Process inputs
        for spec in self.input_specs:
            name = spec["name"]
            data = kwargs[name]

            # Validate and convert dtype if needed
            if data.dtype != spec["dtype"]:
                data = data.astype(spec["dtype"])

            # Set input shape for dynamic shapes
            actual_shape = data.shape
            if self.context.get_tensor_shape(name) != actual_shape:
                if not self.context.set_input_shape(name, actual_shape):
                    raise RuntimeError(
                        f"Failed to set shape {actual_shape} for input '{name}'"
                    )

            # Allocate/reallocate buffer if needed
            device_buffer = self._allocate_buffer(name, actual_shape, spec["dtype"])

            # Copy data to device
            data_contiguous = np.ascontiguousarray(data.ravel())
            cuda.memcpy_htod_async(device_buffer, data_contiguous, self.stream)

            # Set tensor address
            self.context.set_tensor_address(name, int(device_buffer))

        # Allocate output buffers
        for spec in self.output_specs:
            name = spec["name"]
            # Get actual output shape (may be dynamic)
            output_shape = self.context.get_tensor_shape(name)

            # Allocate buffer
            device_buffer = self._allocate_buffer(name, output_shape, spec["dtype"])

            # Set tensor address
            self.context.set_tensor_address(name, int(device_buffer))

        # Execute inference
        success = self.context.execute_async_v3(self.stream.handle)
        if not success:
            raise RuntimeError("Inference execution failed")

        # Copy outputs back to host
        output_arrays = []
        for spec in self.output_specs:
            name = spec["name"]
            output_shape = self.context.get_tensor_shape(name)

            # Allocate host memory
            host_arr = np.empty(output_shape, dtype=spec["dtype"])

            # Get device buffer
            device_buffer, _ = self.output_buffers[name]

            # Copy from device to host
            cuda.memcpy_dtoh_async(host_arr, device_buffer, self.stream)
            output_arrays.append(host_arr)

        # Synchronize stream
        self.stream.synchronize()

        return output_arrays

    def infer_dict(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Run inference and return outputs as a dictionary.

        Args:
            **kwargs: input_name -> np.ndarray mappings

        Returns:
            Dict mapping output names to np.ndarray
        """
        output_arrays = self.infer(**kwargs)
        return {
            spec["name"]: arr for spec, arr in zip(self.output_specs, output_arrays)
        }

    def get_input_info(self) -> List[Dict]:
        """Get information about input tensors."""
        return [
            {"name": s["name"], "shape": s["shape"], "dtype": s["dtype"]}
            for s in self.input_specs
        ]

    def get_output_info(self) -> List[Dict]:
        """Get information about output tensors."""
        return [
            {"name": s["name"], "shape": s["shape"], "dtype": s["dtype"]}
            for s in self.output_specs
        ]

    def __del__(self):
        """Cleanup resources."""
        try:
            # Free input buffers
            for buffer, _ in self.input_buffers.values():
                if hasattr(buffer, "free"):
                    buffer.free()

            # Free output buffers
            for buffer, _ in self.output_buffers.values():
                if hasattr(buffer, "free"):
                    buffer.free()
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()


def load_model_onnx(model_name: str):
    # providers = [
    #     (
    #         "TensorrtExecutionProvider",
    #         {
    #             "trt_fp16_enable": False,
    #             "trt_engine_cache_enable": True,
    #             "trt_engine_cache_path": "./trt_cache",
    #         },
    #     )
    # ]

    providers = ["CUDAExecutionProvider"]  

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    ort_session = ort.InferenceSession(
        f"/workspace/src/NaiviBridger/deployment/model_weights/{model_name}.onnx", sess_options, providers=providers
    )

    return ort_session


def load_model_trt(model_name: str):

    trt_model = TRTInfer(model_name)

    return trt_model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes()
    ros_image.step = ros_image.width
    return ros_image


# def to_numpy(tensor):
#     return tensor.cpu().detach().numpy()


def transform_images(
    pil_imgs: List[PILImage.Image],
    image_size: List[int],
    center_crop: bool = False,
    return_img: bool = False,
):
    """Transforms a list of PIL image to a numpy"""

    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = center_crop_pil(
                    pil_img, (h, int(h * IMAGE_ASPECT_RATIO))
                )  # crop to the right ratio
            else:
                pil_img = center_crop_pil(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        if return_img:  # Added for debug purpose on rviz
            return pil_img
        transf_img = transform_numpy(pil_img)
        transf_img = np.expand_dims(transf_img, axis=0)
        transf_imgs.append(transf_img)
    return np.concatenate(transf_imgs, axis=1)


def transform_numpy(image):
    """
    Equivalent to:
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    Args:
        image (np.ndarray): Input image in shape (H, W, 3), values in [0, 255]
    Returns:
        np.ndarray: Normalized image in shape (3, H, W)
    """
    # Convert to float and scale to [0,1]
    # image = image.astype(np.float32) / 255.0

    # Change from HWC to CHW
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = (image - mean) / std

    return image


def center_crop_pil(img, output_size):
    """
    Args:
        img (PIL.Image): Input image
        output_size (tuple): (crop_height, crop_width)
    """
    w, h = img.size
    new_h, new_w = output_size

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom))


# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


# def get_action(diffusion_output, action_stats=ACTION_STATS):
#     # diffusion_output: (B, 2*T+1, 1)
#     # return: (B, T-1)
#     device = diffusion_output.device
#     ndeltas = diffusion_output
#     ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
#     ndeltas = to_numpy(ndeltas)
#     ndeltas = unnormalize_data(ndeltas, action_stats)
#     actions = np.cumsum(ndeltas, axis=1)
#     return from_numpy(actions).to(device)

# def from_numpy(array: np.ndarray) -> torch.Tensor:
#     return torch.from_numpy(array).float()

# def unnormalize_data(ndata, stats):
#     ndata = (ndata + 1) / 2
#     data = ndata * (stats['max'] - stats['min']) + stats['min']
#     return data