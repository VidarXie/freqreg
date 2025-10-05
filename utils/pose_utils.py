import torch


class Pose:
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self, R=None, t=None):
        # construct a camera pose from the given R and/or t
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        # invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

    def to_matrix(self, pose):
        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose)

        batch_shape = pose.shape[:-2]
        device = pose.device
        dtype = pose.dtype

        # Create bottom row [0, 0, 0, 1]
        bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
        bottom_row = bottom_row.view(*([1] * len(batch_shape)), 1, 4)
        bottom_row = bottom_row.expand(*batch_shape, 1, 4)

        # Concatenate to get 4x4 matrix
        matrix_4x4 = torch.cat([pose, bottom_row], dim=-2)
        return matrix_4x4

    def from_matrix(self, matrix):
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)

        assert matrix.shape[-2:] == (4, 4), (
            f"Expected 4x4 matrix, got {matrix.shape[-2:]}"
        )

        # Extract the [3,4] portion (R|t)
        pose = matrix[..., :3, :4]
        return pose


POSE_ = Pose()
