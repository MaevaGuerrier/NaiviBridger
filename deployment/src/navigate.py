import argparse
import os

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


from vint_train.models.model_utils import create_noise_scheduler
from vint_train.training.train_utils import get_action
from vint_train.visualizing.action_utils import plot_trajs_and_points
from vint_train.models.navibridge.ddbm.karras_diffusion import karras_sample

import torch
import numpy as np
import yaml
from PIL import Image as PILImage
import matplotlib.pyplot as plt

from utils import to_numpy, transform_images, load_model, to_numpy, transform_images, pil_to_msg, msg_to_pil

import time

# ROS 

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import Bool, Float32MultiArray, Int32
from nav_msgs.msg import Path


from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC,
                        CLOSEST_NODE_TOPIC)



TOPOMAP_IMAGES_DIR = "../topomaps/images"
PARAMS_PATH = "../config/params.yaml"
with open(PARAMS_PATH, "r") as f:
    params_config = yaml.safe_load(f)
# CONSTANTS
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 
ACTION_STATS = {}
ACTION_STATS['min'] = np.array([-2.5, -4])
ACTION_STATS['max'] = np.array([5, 4])

# GLOBALS
context_queue = []
context_size = None
subgoal = []

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

def get_bottom_folder_name(path):
    folder_path = os.path.dirname(path)
    bottom_folder_name = os.path.basename(folder_path)
    return bottom_folder_name

def ensure_directory_exists(save_path):
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# def path_project_plot(img, path, args, camera_extrinsics, camera_intrinsics):
#     for i, naction in enumerate(path):
#         gc_actions = to_numpy(get_action(naction))
#         fig = project_and_draw(img, gc_actions, camera_extrinsics, camera_intrinsics)
#         dir_basename = get_bottom_folder_name(image_path)
#         save_path = os.path.join('../output', dir_basename, f'png_{args.model}_image_with_trajs_{i}.png')
#         ensure_directory_exists(save_path)
#         fig.savefig(save_path)
#         save_path = os.path.join('../output', dir_basename, f'svg_{args.model}_image_with_trajs_{i}.svg')
#         ensure_directory_exists(save_path)
#         fig.savefig(save_path)
#         print(f"output image saved as {save_path}")

def main(args):
    camera_intrinsics = np.array([[470.7520828622471, 0, 16.00531005859375],
                    [0, 470.7520828622471, 403.38909912109375],
                    [0, 0, 1]])
    camera_extrinsics = np.array([[0, 0, 1, -0.600],
                                  [-1, 0, 0, -0.000],
                                  [0, -1, 0, -0.042],
                                  [0, 0, 0, 1]])
    global context_size
    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
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
    ckpth_path = model_paths[args.model]["ckpt_path"]
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

    # print(model.__dict__.keys())
    # exit()

     # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    reached_goal = False

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    waypoint_viz_pub = rospy.Publisher(
        "viz_wp", PoseStamped, queue_size=1)
    path_viz_pub = rospy.Publisher(
        "viz_path", Path, queue_size=1)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    distances_pub = rospy.Publisher("/distances", Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
    goal_img_pub = rospy.Publisher("/topoplan/goal_img", Image, queue_size=1)
    subgoal_img_pub = rospy.Publisher("/topoplan/subgoal_img", Image, queue_size=1)
    closest_node_img_pub = rospy.Publisher("/topoplan/closest_node_img", Image, queue_size=1)
    closest_node_pub = rospy.Publisher(CLOSEST_NODE_TOPIC, Int32, queue_size=10)



    if model_params["model_type"] == "navibridge":
        noise_scheduler, diffusion = create_noise_scheduler(model_params)
    

    while not rospy.is_shutdown():
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            # obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            # obs_images = obs_images.to(device)
            # mask = torch.ones(1).long().to(device)
            time_0 = time.time()
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            # import pdb; pdb.set_trace()
            # obs_images = torch.split(obs_images, 3, dim=1)
            # obs_images = torch.cat(obs_images, dim=1) 
            obs_images = obs_images.to(device)
            mask = torch.zeros(1).long().to(device) 


            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
            goal_image = torch.concat(goal_image, dim=0)
            goal_image = goal_image.to(device)

            obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
            
            dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            distances_msg = Float32MultiArray()
            distances_msg.data = dists
            distances_pub.publish(distances_msg)
            

            min_idx = np.argmin(dists)
            closest_node = min_idx + start
            closest_node_msg = Int32()
            closest_node_msg.data = closest_node
            closest_node_pub.publish(closest_node_msg)
            
            sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
            obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
            print(f"start {start} clnod {closest_node} sg_idx {sg_idx} dis {dists} ")


            with torch.no_grad():
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                if model_params["model_type"] == "navibridge":
                    if model_params["prior_policy"] == "handcraft":
                        # Predict aciton states
                        states_pred = model("states_pred_net", obsgoal_cond=obs_cond)

                    if model_params["prior_policy"] == "cvae":
                        prior_cond = obs_images.repeat_interleave(args.num_samples, dim=0)
                    elif model_params["prior_policy"] == "handcraft":
                        prior_cond = states_pred
                    else:
                        prior_cond = None

                    # initialize action from Gaussian noise
                    if model.prior_model.prior is None:
                        initial_samples = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    else:
                        with torch.no_grad():
                            initial_samples = model.prior_model.sample(cond=prior_cond, device=device)
                        assert initial_samples.shape[-1] == 2, "action dim must be 2"
                    time_diffusion_start = time.time()
                    naction, path, nfe = karras_sample(
                        diffusion,
                        model,
                        initial_samples,
                        None,
                        steps=model_params["num_diffusion_iters"],
                        model_kwargs=initial_samples,
                        global_cond=obs_cond,
                        device=device,
                        clip_denoised=model_params["clip_denoised"],
                        sampler="heun",
                        sigma_min=diffusion.sigma_min,
                        sigma_max=diffusion.sigma_max,
                        churn_step_ratio=model_params["churn_step_ratio"],
                        rho=model_params["rho"],
                        guidance=model_params["guidance"]
                    )
                    print(f"Diffusion time: {time.time() - time_diffusion_start} sec")
                    print(f"Inference time: {time.time() - time_0} sec")

            # if args.path_visual:
            #     path_project_plot(context_queue[-1], path, args, camera_extrinsics, camera_intrinsics)

            naction = to_numpy(get_action(naction))
            # @TODO what is this so 3xmax_v first then divide? from naivirbidge codebase
            # scale_factor= 3 * MAX_V / RATE
            print(f"MAX V: {MAX_V}, RATE: {RATE}")
            scale_factor= (MAX_V / RATE)
            naction *= scale_factor

            sampled_actions_msg = Float32MultiArray()
            sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
            sampled_actions_pub.publish(sampled_actions_msg)
            naction = naction[0] 
            chosen_waypoint = naction[args.waypoint]

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)

        reached_goal = closest_node == goal_node
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached goal! Stopping...")
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run DIFFUSION NAVIGATION demo")
    parser.add_argument(
        "--model",
        "-m",
        default="navibridger_cvae",
        type=str,
        help="model name (hint: check ../config/models.yaml)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8, #code base naivibridge was 100
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--path-visual",
        default=False,
        type=bool,
        help="visualization",
    )

    parser.add_argument(
        "--dir",
        "-d",
        default="sim_test",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=0.5,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=2,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    args = parser.parse_args()
    print(f"Using model {args.model} on device {device} ")
    main(args)
