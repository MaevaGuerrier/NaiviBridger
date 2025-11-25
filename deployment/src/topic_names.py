# topic names for ROS communication
# IMPORTANT THIS FILE HAS TO BE IN THE SAME FOLDER AS SHELL SCRIPTS (record_bag.sh, explore.sh, navigate.sh)

# Launch file related information for your robot 

# SIMULATION ONLY 
# LAUNCH_FILE="turtle_bot.launch" 
# LAUNCH_PKG = "limo_gazebo_sim"

# REAL ROBOT 
LAUNCH_FILE = "bunker_GNM.launch"
LAUNCH_PKG = "rover_launch/launch/SafeGNM"  


TELEOP_PKG = "teleop_twist_keyboard"
TELEOP_LAUNCH = "teleop_twist_keyboard.py"

# image obs topics
FRONT_IMAGE_TOPIC = "/usb_cam_front/image_raw" 
REVERSE_IMAGE_TOPIC = "/usb_cam_reverse/image_raw"
IMAGE_TOPIC = "/usb_cam/image_raw" # oak lite pro /oak/rgb/image_raw # d400 "/d400_arm/color/image_raw" # fisheye "/usb_cam/image_raw"
CLOSEST_NODE_TOPIC = "/closest_node"

# exploration topics
SUBGOALS_TOPIC = "/subgoals"
GRAPH_NAME_TOPIC = "/graph_name"
WAYPOINT_TOPIC = "/waypoint"
REVERSE_MODE_TOPIC = "/reverse_mode"
SAMPLED_OUTPUTS_TOPIC = "/sampled_outputs"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"
SAMPLED_WAYPOINTS_GRAPH_TOPIC = "/sampled_waypoints_graph"
BACKTRACKING_IMAGE_TOPIC = "/backtracking_image"
FRONTIER_IMAGE_TOPIC = "/frontier_image"
SUBGOALS_SHAPE_TOPIC = "/subgoal_shape"
SAMPLED_ACTIONS_TOPIC = "/sampled_actions"
ANNOTATED_IMAGE_TOPIC = "/annotated_image"
CURRENT_NODE_IMAGE_TOPIC = "/current_node_image"
FLIP_DIRECTION_TOPIC = "/flip_direction"
TURNING_TOPIC = "/turning"
SUBGOAL_GEN_RATE_TOPIC = "/subgoal_gen_rate"
MARKER_TOPIC = "/visualization_marker_array"
VIZ_NAV_IMAGE_TOPIC = "/nav_image"

# visualization topics
CHOSEN_SUBGOAL_TOPIC = "/chosen_subgoal"

# recorded ont the robot
ODOM_TOPIC = "/odom"
BUMPER_TOPIC = "/mobile_base/events/bumper"
JOY_BUMPER_TOPIC = "/joy_bumper"

# move the robot