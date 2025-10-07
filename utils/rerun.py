import rerun as rr
import rerun.blueprint as rrb
import torch
import numpy as np


def create_blueprint(parent_log_path):
    """
    Create a blueprint for the rerun visualization.
    """
    blueprint = rrb.Blueprint(rrb.Spatial3DView(origin=parent_log_path))
    return blueprint


class RerunLogger:
    """
    Logger for rerun visualization.
    """

    def __init__(self, parent_log_path):
        self.parent_log_path = parent_log_path
        rr.log(str(self.parent_log_path), rr.ViewCoordinates.RDF, static=True)

    def log_poses_at_frame(self, gt_poses, pred_poses, frame_id):
        """
        Log both ground truth and predicted poses at the same frame.
        This ensures all camera data appears simultaneously.
        """
        # Set the frame for all subsequent logging
        rr.set_time_sequence("frame", frame_id)

        # Log ground truth poses
        gt_log_path = self.parent_log_path / "gt_poses"

        # Extract camera centers using the same method as get_camera_mesh
        # Camera center is at (0, 0, 0) in camera coordinates
        center = torch.zeros(1, 1, 3, device=gt_poses.device)
        gt_centers = camera.cam2world(center, gt_poses)[:, 0]  # Shape: (N, 3)
        gt_centers_np = gt_centers.detach().cpu().numpy()
        gt_rotations_np = gt_poses[..., :3, :3].transpose(-2, -1).detach().cpu().numpy()

        # Log all ground truth camera centers
        rr.log(
            f"{gt_log_path}/centers",
            rr.Points3D(
                positions=gt_centers_np,
                colors=[0, 0, 255],
                radii=0.05,  # Make points more visible
            ),
        )

        gt_cam_log_path = self.parent_log_path / "gt_poses" / "cameras"

        line_stips = self._draw_camera_frustum(0.2, ratio=1.0)

        for i, (gt_rot, gt_cen) in enumerate(zip(gt_rotations_np, gt_centers_np)):
            rr.log(
                f"{gt_cam_log_path}/{i}",
                rr.Transform3D(
                    translation=gt_cen,
                    mat3x3=gt_rot,
                ),
            )
            rr.log(
                f"{gt_cam_log_path}/{i}/frustum",
                rr.LineStrips3D(strips=line_stips, colors=[0, 0, 255], radii=0.01),
            )

        # Log predicted poses
        pred_log_path = self.parent_log_path / "pred_poses"

        # Extract camera centers for predicted poses
        pred_centers = camera.cam2world(center, pred_poses)[:, 0]  # Shape: (N, 3)
        pred_centers_np = pred_centers.detach().cpu().numpy()
        pred_rotations_np = (
            pred_poses[..., :3, :3].transpose(-2, -1).detach().cpu().numpy()
        )

        # Log all predicted camera centers
        rr.log(
            f"{pred_log_path}/centers",
            rr.Points3D(
                positions=pred_centers_np,
                colors=[255, 0, 0],
                radii=0.05,  # Make points more visible
            ),
        )

        pred_cam_log_path = self.parent_log_path / "pred_poses" / "cameras"

        for i, (pred_rot, pred_cen) in enumerate(
            zip(pred_rotations_np, pred_centers_np)
        ):
            rr.log(
                f"{pred_cam_log_path}/{i}",
                rr.Transform3D(
                    translation=pred_cen,
                    mat3x3=pred_rot,
                ),
            )
            rr.log(
                f"{pred_cam_log_path}/{i}/frustum",
                rr.LineStrips3D(strips=line_stips, colors=[255, 0, 0], radii=0.01),
            )

        # Log line segments connecting corresponding GT and predicted camera centers
        error_log_path = self.parent_log_path / "pose_errors"

        # Create line segments from GT to predicted centers
        for i, (gt_center, pred_center) in enumerate(
            zip(gt_centers_np, pred_centers_np)
        ):
            # Create a line segment using LineStrips3D
            line_points = np.array([gt_center, pred_center])
            rr.log(
                f"{error_log_path}/connection_{i}",
                rr.LineStrips3D(
                    strips=line_points,
                    colors=[250, 128, 114],  # Salmon color for errors
                ),
            )

    def _draw_camera_frustum(self, width, ratio=1080 / 1920, z_ratio=0.8):
        w = width
        h = w * ratio
        z = w * z_ratio

        frustum_line = [
            [[0, 0, 0], [w, h, z]],
            [[0, 0, 0], [w, -h, z]],
            [[0, 0, 0], [-w, -h, z]],
            [[0, 0, 0], [-w, h, z]],
            [[w, h, z], [w, -h, z]],
            [[-w, h, z], [-w, -h, z]],
            [[-w, h, z], [w, h, z]],
            [[-w, -h, z], [w, -h, z]],
        ]

        line_stips = []
        for line in frustum_line:
            line_stips.extend([line[0], line[1]])

        return line_stips
