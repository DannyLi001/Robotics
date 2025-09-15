from ..config.config import get_rpc_classes
from ..server.arm_server.arm_server import ArmManager
from ..server.base_server.base_server import BaseManager
import sys
import os
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

RPC_CFG = get_rpc_classes()

BASE_RPC_HOST = RPC_CFG['base'].host
BASE_RPC_PORT = RPC_CFG['base'].port
authkey_str = RPC_CFG['base'].authkey
BASE_RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str

ARM_RPC_HOST = RPC_CFG['arm'].host
ARM_RPC_PORT = RPC_CFG['arm'].port
authkey_str = RPC_CFG['arm'].authkey
ARM_RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str

class RealEnv:
    def __init__(
        self
    ):
        # RPC server connection for base
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception(f'Could not connect to base RPC server, is base_server.py running? {e}\n [IMPORTANT]: sudo netplan apply') from e

        # RPC server connection for arm
        arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e

        # RPC proxy objects
        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        self.arm = arm_manager.Arm()

        # Camera name 
        self.cameras = {}

        for cam_cfg in self.cfg.cameras:
            cam_name = cam_cfg.name
            cam_type = cam_cfg.type.lower()
            if cam_type == "realsense":
                assert cam_cfg.serial is not None, f"Camera '{cam_name}' of type 'realsense' must specify a serial"
                cam = RealSenseCamera(cam_cfg.serial, frame_width=640, frame_height=480, fps=30, use_depth=self.use_depth)
            else:
                raise ValueError(f"Unknown camera type: {cam_type}")

            self.cameras[cam_name] = cam

    def get_state_obs(self):
        obs = dict()
        obs.update(self.base.get_state())
        obs.update(self.arm.get_state())
        return obs

    def get_cam_obs(self):
        obs = dict()
        for name, cam in self.cameras.items():
            if self.use_depth:
                rgb_image, depth_image = cam.get_rgbd()
                obs.update({f"{name}_image": rgb_image, f"{name}_depth": depth_image})
            else:
                obs.update({f"{name}_image": cam.get_image()})
        return obs

    def get_obs(self):
        obs = self.get_state_obs()
        obs.update(self.get_cam_obs())
        return obs

    def reset(self):
        print('Resetting base...')
        self.base.reset()

        print('Resetting arm...')
        self.arm.reset()

        print('Robot has been reset')
        
        return self.get_obs()

    def step(self, action):
        self.base.execute_action(action['base'])
        self.arm.execute_action(action['arm'])
        
    def step_arm_only(self, action):
        self.arm.execute_action(action)

    def step_base_only(self, action):
        self.base.execute_action(action)

    def set_gripper(self, width):
        obs = self.get_obs()
        pos = obs['arm_pos']
        euler = R.from_quat(obs['arm_quat']).as_euler('xyz')
        action = {
            'arm_pos': pos,
            'arm_euler': euler,
            'gripper_pos': np.array([width]),
            'grasp': False
        }
        self.arm.execute_action(action)

    def close(self):
        self.base.close()
        self.arm.close()
        for cam in self.cameras.values():
            cam.close()


class RealPointCloudEnv:
    def __init__(
        self,
        env: 'RealBaseEnv',
        num_objects: int = 2,
        device: str = 'cuda',
        sam2_path: str = "/home/lijiawei/VLA/internavi/REALTIME_SAM2",
        sam2_checkpoint: str = "/home/lijiawei/VLA/internavi/REALTIME_SAM2/checkpoints/sam2.1_hiera_tiny.pt",
        sam2_config: str = "sam2.1/sam2.1_hiera_t"
    ):
        """
        A wrapper environment that adds SAM2 tracking and pointcloud generation capabilities.

        Args:
            env (RealBaseEnv): The underlying robot environment instance.
            num_objects (int): The number of objects to track simultaneously.
            device (str): Device to run SAM2 on.
            sam2_path (str): Path to SAM2 repository.
            sam2_checkpoint (str): Path to SAM2 checkpoint file.
            sam2_config (str): SAM2 model configuration name.
        """
        self._env = env
        self.num_objects = num_objects
        self.classes = list(range(self.num_objects))
        self.device = torch.device(device)
        self.sam2_path = sam2_path
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        
        # Setup SAM2
        if self.sam2_path not in sys.path:
            sys.path.insert(0, self.sam2_path)
        
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        if not GlobalHydra.instance().is_initialized():
            config_dir = os.path.join(self.sam2_path, "sam2", "configs")
            initialize_config_dir(config_dir=config_dir, version_base="1.2")

        from REALTIME_SAM2.sam2.build_sam import build_sam2_camera_predictor
        self.predictor = build_sam2_camera_predictor(self.sam2_config, self.sam2_checkpoint, device=self.device)
        
        # Internal state to store the output from SAM2
        self.out_mask_logits = None
        self.initial_points = None
        
        # Ensure at least one camera is configured
        if not self._env.cfg.cameras:
            raise ValueError("The environment must have at least one camera configured.")
        self.main_camera_name = self._env.cfg.cameras[0].name
        
        # Get camera intrinsics for 3D projection
        main_camera = self._env.cameras[self.main_camera_name]
        self.intrinsics = main_camera.get_intrinsics()
        self.depth_scale = self.intrinsics['depth_scale']
        self.K = self.intrinsics['matrix']
        
        # Create RealSense intrinsics object for deprojection
        self.rs_intrinsics = rs.intrinsics()
        self.rs_intrinsics.width = self.intrinsics['width']
        self.rs_intrinsics.height = self.intrinsics['height']
        self.rs_intrinsics.fx = self.K[0, 0]
        self.rs_intrinsics.fy = self.K[1, 1]
        self.rs_intrinsics.ppx = self.K[0, 2]
        self.rs_intrinsics.ppy = self.K[1, 2]
        self.rs_intrinsics.model = rs.distortion.none
        self.rs_intrinsics.coeffs = [0, 0, 0, 0, 0]

    def _get_binary_masks(self) -> list:
        """Convert mask logits to binary numpy masks."""
        if self.out_mask_logits is None:
            return [np.zeros((480, 640), dtype=np.uint8)] * self.num_objects
        
        masks_np = self.out_mask_logits.cpu().numpy()
        binary_masks = []
        for i in range(self.num_objects):
            mask = (masks_np[i] > 0.0).astype(np.uint8)
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask[0]
            binary_masks.append(mask)
        return binary_masks

    def _mask_to_pointcloud(self, mask, depth_image, object_type):
        """Convert 2D mask to 3D pointcloud using RealSense SDK."""
        # Get mask coordinates
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return np.array([]).reshape(0, 4)  # Empty pointcloud
        
        # Get depth values at mask coordinates
        if len(depth_image.shape) == 3:
            depths = depth_image[y_coords, x_coords, 0] * self.depth_scale
        else:
            depths = depth_image[y_coords, x_coords] * self.depth_scale
        
        # Filter out invalid depth values
        valid_mask = (depths > 0) & (depths < 5.0)  # Filter reasonable depth range
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        depths = depths[valid_mask]
        
        if len(x_coords) == 0:
            return np.array([]).reshape(0, 4)  # Empty pointcloud
        
        # Use RealSense SDK to convert 2D pixels to 3D points
        points_3d = []
        for i in range(len(x_coords)):
            pixel = [int(x_coords[i]), int(y_coords[i])]
            depth = depths[i]
            
            # Use RealSense SDK's built-in deprojection
            point_3d = rs.rs2_deproject_pixel_to_point(self.rs_intrinsics, pixel, depth)
            points_3d.append(point_3d)
        
        # Convert to numpy array and add type information
        points_3d = np.array(points_3d)
        pointcloud = np.column_stack([points_3d, np.full(len(points_3d), object_type)])
        
        return pointcloud

    def get_segmented_pointcloud(self, rgb_image, depth_image, binary_masks):
        """Get segmented pointcloud for all objects."""
        all_pointclouds = []
        
        for i, mask in enumerate(binary_masks):
            if mask.sum() > 0:  # Only process non-empty masks
                pointcloud = self._mask_to_pointcloud(mask, depth_image, i)
                if len(pointcloud) > 0:
                    all_pointclouds.append(pointcloud)
        
        if all_pointclouds:
            return np.vstack(all_pointclouds)
        else:
            return np.array([]).reshape(0, 4)

    def reset(self, initial_points: list):
        """
        Resets the robot and the tracker. Initial points must be provided to start tracking.

        Args:
            initial_points (list): A list of [x, y] coordinates for each object.

        Returns:
            dict: The initial observation dictionary, augmented with pointcloud data.
        """
        if len(initial_points) != self.num_objects:
            raise ValueError(f"Expected {self.num_objects} initial points, but received {len(initial_points)}")

        self.initial_points = initial_points
        
        # Reset the underlying robot environment
        obs = self._env.reset()
        
        # Get the first frame and depth
        frame = obs[f"{self.main_camera_name}_image"]
        depth_image = obs[f"{self.main_camera_name}_depth"]
        
        # Load first frame into SAM2
        self.predictor.load_first_frame(frame, self.num_objects)
        
        # Add initial points to SAM2
        first_hit = np.array([True], dtype=np.bool_)
        for i, point in enumerate(initial_points):
            obj_id = self.classes[i]
            points_np = np.array([point], dtype=np.float32)
            labels_np = np.array([1], dtype=np.int32)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                self.predictor.add_new_prompt_during_track(
                    obj_id, points_np, labels_np, first_hit=first_hit[0], frame=frame
                )
            first_hit = np.array([False], dtype=np.bool_)
        
        # Get initial masks
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            _, _, self.out_mask_logits = self.predictor.finalize_new_input()

        return self.get_obs()

    def step(self, action):
        """
        Executes an action in the underlying environment, then updates the SAM2 tracking.

        Args:
            action (dict): The action dictionary for the robot's base and/or arm.

        Returns:
            dict: The new observation dictionary after the step, augmented with pointcloud data.
        """
        # Execute the action in the underlying environment
        self._env.step(action)
        
        # Get the new image frame after the action is executed
        obs_after_step = self._env.get_cam_obs()
        frame = obs_after_step[f"{self.main_camera_name}_image"]

        # Perform tracking on the new frame using SAM2
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            _, self.out_mask_logits = self.predictor.track(frame)
            
        return self.get_obs()
        
    def get_obs(self):
        """
        Gets the observation from the underlying environment and adds the pointcloud data.
        
        Returns:
            dict: The augmented observation dictionary with pointcloud data.
        """
        # Get robot state and camera images from the underlying environment
        obs = self._env.get_obs()
        
        # Get binary masks and generate pointcloud
        binary_masks = self._get_binary_masks()
        rgb_image = obs[f"{self.main_camera_name}_image"]
        depth_image = obs[f"{self.main_camera_name}_depth"]
        
        # Generate segmented pointcloud
        pointcloud = self.get_segmented_pointcloud(rgb_image, depth_image, binary_masks)
        
        # Add pointcloud to observation
        obs['pointcloud'] = pointcloud
        obs['binary_masks'] = binary_masks
        
        return obs


class RealPointTrackEnv:
    def __init__(
        self,
        env,
        task_name,
        object_labels,
        calib_path,
        max_episode_len=300,
        max_state_dim=100,
        pixel_keys=None,
        points_cfg=None,
        point_dim=2,
        device='cpu',
        return_2d_points=False
    ):
        self._env = env
        self._task_name = task_name
        self._object_labels = object_labels
        self._max_episode_len = max_episode_len
        self._max_state_dim = max_state_dim
        self._pixel_keys = pixel_keys
        self._device = device
        self._use_gt_depth = self._env.use_depth
        self._point_dim = point_dim
        self._return_2d_points = return_2d_points

        points_cfg["task_name"] = task_name
        points_cfg["pixel_keys"] = self._pixel_keys
        points_cfg["object_labels"] = object_labels
        self._points_class = PointsClass(**points_cfg)

        # calibration data
        assert calib_path is not None
        self.calibration_data = np.load(calib_path, allow_pickle=True).item()
        self._camera_names = list(self.calibration_data.keys())
        self.camera_projections = {}
        for camera_name in self._camera_names:
            intrinsic = self.calibration_data[camera_name]["int"]
            intrinsic = np.concatenate((intrinsic, np.zeros((3, 1))), axis=1)
            extrinsic = self.calibration_data[camera_name]["ext"]
            extrinsic = np.linalg.inv(extrinsic)
            self.camera_projections[camera_name] = intrinsic @ extrinsic

    def reset(self):
        obs = self._env.reset()
        arm_qpos = self._env.get_state_obs()['arm_qpos'].astype(np.float32)
        arm_qvel = self._env.get_state_obs()['arm_qvel'].astype(np.float32)
        gripper_pos = self._env.get_state_obs()['gripper_pos'].astype(np.float32)
        gripper_vel = self._env.get_state_obs()['gripper_vel'].astype(np.float32)

        arm_pos = self._env.get_state_obs()['arm_pos'].astype(np.float32)
        arm_quat = self._env.get_state_obs()['arm_quat'].astype(np.float32)

        observation = dict(
            arm_qpos=arm_qpos,
            arm_qvel=arm_qvel,
            gripper_pos=gripper_pos,
            gripper_vel=gripper_vel,
            arm_pos=arm_pos,
            arm_quat=arm_quat,
            goal_pose=np.array([0, 0, 0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32) # update in policy
        )

        # point tracker init
        self.init_track_points(obs)

        for cam in self._env.cfg.cameras:
            if self._return_2d_points:
                observation.update({f"point_tracks_{cam.name}_2d": self._track_pts[cam.name]})

            pts = self._track_pts[cam.name].astype(np.float32).reshape(-1, 1, 2)
            intr = self.calibration_data[cam.name]["int"]
            dist = self.calibration_data[cam.name]["dist_coeff"]
            pts_undistorted = cv2.undistortPoints(pts, intr, dist, None, intr).reshape(-1, 2)
            
            self._track_pts[cam.name] = pts_undistorted

        # Get 3d points from 3D depth or 2D triangulation
        if self._point_dim == 3:
            if not self._use_gt_depth:
                P, pts = [], []
                for cam in self._env.cfg.cameras:
                    camera_name = cam.name
                    P.append(self.camera_projections[camera_name])
                    pt2d = self._track_pts[cam.name]
                    pts.append(pt2d)

                pts3d = triangulate_points(P, pts)[:, :3]
                for cam in self._env.cfg.cameras:
                    observation.update({f"point_tracks_3d": np.array(pts3d, dtype=np.float32)})
            else:
                for cam in self._env.cfg.cameras:
                    camera_name = cam.name
                    pt2d = self._track_pts[cam.name]
                    depth_key = f"{cam.name}_depth"
                    depth = obs[depth_key]
                    # compute depth for each points
                    depths = []
                    for pt in pt2d:
                        x, y = pt.astype(int)
                        depths.append(depth[y, x].item())
                    depths = np.array(depths) / 1000.0  # convert to meters
                    extr = self.calibration_data[camera_name]["ext"]
                    intr = self.calibration_data[camera_name]["int"]
                    pt3d = pixel2d_to_3d(pt2d, depths, intr, extr)
                    observation.update({f"point_tracks_{cam.name}_3d": pt3d})

        return observation

    def step(self, action):
        self._env.step(action)

    def get_obs(self):
        obs = self._env.get_obs()
        arm_qpos = self._env.get_state_obs()['arm_qpos'].astype(np.float32)
        arm_qvel = self._env.get_state_obs()['arm_qvel'].astype(np.float32)
        gripper_pos = self._env.get_state_obs()['gripper_pos'].astype(np.float32)
        gripper_vel = self._env.get_state_obs()['gripper_vel'].astype(np.float32)

        arm_pos = self._env.get_state_obs()['arm_pos'].astype(np.float32)
        arm_quat = self._env.get_state_obs()['arm_quat'].astype(np.float32)

        observation = dict(
            arm_qpos=arm_qpos,
            arm_qvel=arm_qvel,
            gripper_pos=gripper_pos,
            gripper_vel=gripper_vel,
            arm_pos=arm_pos,
            arm_quat=arm_quat,
            goal_pose=np.array([0, 0, 0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32) # update in policy
        )

        for cam in self._env.cfg.cameras:
            self._points_class.add_to_image_list(
                obs[f"{cam.name}_image"][:, :, ::-1], cam.name
            )
            self._points_class.track_points(cam.name)
            object_pts = self._points_class.get_points_on_image(cam.name).cpu().numpy()[
                0
            ]

            if self._return_2d_points:
                observation.update({f"point_tracks_{cam.name}_2d": object_pts})

            pts = object_pts.astype(np.float32).reshape(-1, 1, 2)
            intr = self.calibration_data[cam.name]["int"]
            dist = self.calibration_data[cam.name]["dist_coeff"]
            pts_undistorted = cv2.undistortPoints(pts, intr, dist, None, intr).reshape(-1, 2)
            
            self._track_pts[cam.name] = pts_undistorted

        # Get 3d points from 3D depth or 2D triangulation
        if self._point_dim == 3:
            if not self._use_gt_depth:
                P, pts = [], []
                for cam in self._env.cfg.cameras:
                    camera_name = cam.name
                    P.append(self.camera_projections[camera_name])
                    pt2d = self._track_pts[cam.name]
                    pts.append(pt2d)

                pts3d = triangulate_points(P, pts)[:, :3]
                observation.update({f"point_tracks_3d": np.array(pts3d, dtype=np.float32)})
            else:
                for cam in self._env.cfg.cameras:
                    camera_name = cam.name
                    pt2d = self._track_pts[cam.name]
                    depth_key = f"{cam.name}_depth"
                    depth = obs[depth_key]
                    # compute depth for each points
                    depths = []
                    for pt in pt2d:
                        x, y = pt.astype(int)
                        depths.append(depth[y, x].item())
                    depths = np.array(depths) / 1000.0  # convert to meters
                    extr = self.calibration_data[camera_name]["ext"]
                    intr = self.calibration_data[camera_name]["int"]
                    pt3d = pixel2d_to_3d(pt2d, depths, intr, extr)
                    observation.update({f"point_tracks_{cam.name}_3d": pt3d})

        return observation

    def init_track_points(self, obs):
        self._track_pts = {}
        for cam in self._env.cfg.cameras:
            points = []
            frame = obs[f"{cam.name}_image"]
            self._points_class.reset_episode()
            self._points_class.add_to_image_list(frame[:, :, ::-1], cam.name)
            for object_label in self._object_labels:
                self._points_class.find_semantic_similar_points(
                    cam.name, object_label
                )
            self._points_class.track_points(cam.name, is_first_step=True)
            self._points_class.track_points(cam.name)
            # object_pts = self._points_class.get_points_on_image(cam.name)
            object_pts = self._points_class.get_points_on_image(cam.name).to(self._device)
            points.append(object_pts)
            
            self._track_pts[cam.name] = torch.cat(points, dim=1)[0].numpy()