"""Bundle adjustment for Assignment 3.

This script optimizes a shared focal length, per-view camera extrinsics, and
3D point coordinates from the provided 2D observations.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


IMAGE_SIZE = 1024
CX = IMAGE_SIZE / 2.0
CY = IMAGE_SIZE / 2.0


def euler_xyz_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles in radians to rotation matrices."""
    x, y, z = angles.unbind(dim=-1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rx = torch.stack(
        [
            ones,
            zeros,
            zeros,
            zeros,
            cx,
            -sx,
            zeros,
            sx,
            cx,
        ],
        dim=-1,
    ).reshape(*angles.shape[:-1], 3, 3)
    ry = torch.stack(
        [
            cy,
            zeros,
            sy,
            zeros,
            ones,
            zeros,
            -sy,
            zeros,
            cy,
        ],
        dim=-1,
    ).reshape(*angles.shape[:-1], 3, 3)
    rz = torch.stack(
        [
            cz,
            -sz,
            zeros,
            sz,
            cz,
            zeros,
            zeros,
            zeros,
            ones,
        ],
        dim=-1,
    ).reshape(*angles.shape[:-1], 3, 3)
    return rz @ ry @ rx


def project(points: torch.Tensor, euler: torch.Tensor, trans: torch.Tensor, focal: torch.Tensor) -> torch.Tensor:
    """Project world points to all cameras.

    Args:
        points: (B, 3) 3D points.
        euler: (V, 3) camera Euler angles.
        trans: (V, 3) camera translations.
        focal: scalar focal length in pixels.

    Returns:
        (V, B, 2) pixel coordinates.
    """
    rot = euler_xyz_to_matrix(euler)
    cam = torch.einsum("vij,bj->vbi", rot, points) + trans[:, None, :]
    z = cam[..., 2].clamp(max=-1.0e-4)
    u = -focal * cam[..., 0] / z + CX
    v = focal * cam[..., 1] / z + CY
    return torch.stack([u, v], dim=-1)


def load_observations(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    data = np.load(path)
    keys = sorted(data.files)
    obs_np = np.stack([data[k][:, :2] for k in keys], axis=0).astype(np.float32)
    vis_np = np.stack([data[k][:, 2] > 0.5 for k in keys], axis=0)
    return (
        torch.from_numpy(obs_np).to(device),
        torch.from_numpy(vis_np).to(device),
        keys,
    )


def inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def initialize(obs: torch.Tensor, vis: torch.Tensor, device: torch.device, focal0: float, distance: float):
    num_views, num_points = vis.shape

    weights = vis.float()
    denom = weights.sum(dim=0).clamp_min(1.0)
    mean_xy = (obs * weights[..., None]).sum(dim=0) / denom[:, None]

    x = (mean_xy[:, 0] - CX) * distance / focal0
    y = -(mean_xy[:, 1] - CY) * distance / focal0
    z = 0.05 * torch.randn(num_points, device=device)
    points = torch.stack([x, y, z], dim=-1)

    euler = torch.zeros((num_views, 3), device=device)
    euler[:, 1] = torch.linspace(
        math.radians(70.0),
        math.radians(-70.0),
        num_views,
        device=device,
    )

    trans = torch.zeros((num_views, 3), device=device)
    trans[:, 2] = -distance

    min_focal = 300.0
    raw_focal = torch.tensor(
        inverse_softplus(focal0 - min_focal),
        dtype=torch.float32,
        device=device,
    )
    return points, euler, trans, raw_focal


def save_obj(path: Path, points: np.ndarray, colors_path: Path) -> None:
    colors = np.load(colors_path).astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    with path.open("w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            f.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def save_loss_csv(path: Path, history: list[tuple[int, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "focal"])
        writer.writerows(history)


def save_loss_svg(path: Path, history: list[tuple[int, float, float]]) -> None:
    width, height = 900, 420
    margin = 48
    losses = np.array([row[1] for row in history], dtype=np.float64)
    steps = np.array([row[0] for row in history], dtype=np.float64)
    if len(losses) == 0:
        return
    ymin, ymax = float(losses.min()), float(losses.max())
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    x0, x1 = float(steps.min()), float(steps.max())
    if math.isclose(x0, x1):
        x1 = x0 + 1.0

    def sx(x):
        return margin + (x - x0) / (x1 - x0) * (width - 2 * margin)

    def sy(y):
        return height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)

    pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(steps, losses))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>
<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{pts}"/>
<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Bundle Adjustment Loss</text>
<text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="13">step</text>
<text x="14" y="{height/2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 14 {height/2:.1f})">MSE reprojection loss</text>
<text x="{margin}" y="{height-margin+18}" font-family="sans-serif" font-size="11">{int(x0)}</text>
<text x="{width-margin}" y="{height-margin+18}" text-anchor="end" font-family="sans-serif" font-size="11">{int(x1)}</text>
<text x="{margin-8}" y="{sy(ymax)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{ymax:.3g}</text>
<text x="{margin-8}" y="{sy(ymin)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{ymin:.3g}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize bundle adjustment from 2D observations.")
    parser.add_argument("--points2d", type=Path, default=Path("data/points2d.npz"))
    parser.add_argument("--colors", type=Path, default=Path("data/points3d_colors.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ba"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-points", type=int, default=4096)
    parser.add_argument("--lr-points", type=float, default=2.0e-3)
    parser.add_argument("--lr-cameras", type=float, default=5.0e-4)
    parser.add_argument("--lr-focal", type=float, default=2.0e-3)
    parser.add_argument("--focal-init", type=float, default=900.0)
    parser.add_argument("--distance-init", type=float, default=2.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    obs, vis, keys = load_observations(args.points2d, device)
    num_views, num_points = vis.shape
    print(f"Loaded {num_views} views, {num_points} points, {int(vis.sum().item())} visible observations")
    print(f"Optimizing on {device}")

    init_points, init_euler, init_trans, init_raw_focal = initialize(
        obs, vis, device, args.focal_init, args.distance_init
    )
    points = torch.nn.Parameter(init_points)
    euler = torch.nn.Parameter(init_euler)
    trans = torch.nn.Parameter(init_trans)
    raw_focal = torch.nn.Parameter(init_raw_focal)
    min_focal = 300.0

    optimizer = torch.optim.Adam(
        [
            {"params": [points], "lr": args.lr_points},
            {"params": [euler, trans], "lr": args.lr_cameras},
            {"params": [raw_focal], "lr": args.lr_focal},
        ]
    )

    history: list[tuple[int, float, float]] = []
    all_indices = torch.arange(num_points, device=device)
    for step in range(1, args.steps + 1):
        if args.batch_points >= num_points:
            idx = all_indices
        else:
            idx = torch.randint(0, num_points, (args.batch_points,), device=device)

        focal = min_focal + F.softplus(raw_focal)
        pred = project(points[idx], euler, trans, focal)
        target = obs[:, idx]
        mask = vis[:, idx]

        residual = pred - target
        reproj_loss = (residual.square().sum(dim=-1)[mask]).mean()

        centered = points - points.mean(dim=0, keepdim=True)
        center_loss = points.mean(dim=0).square().sum()
        scale_loss = (centered.std(dim=0).mean() - 0.45).square()
        tz_loss = (trans[:, 2].mean() + args.distance_init).square()
        roll_pitch_loss = euler[:, [0, 2]].square().mean()
        loss = reproj_loss + 0.01 * center_loss + 0.01 * scale_loss + 0.001 * tz_loss + 0.0005 * roll_pitch_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            loss_value = float(reproj_loss.detach().cpu())
            focal_value = float(focal.detach().cpu())
            history.append((step, loss_value, focal_value))
            print(f"step {step:5d}/{args.steps}  reproj_mse={loss_value:10.4f}  focal={focal_value:8.2f}")

    with torch.no_grad():
        final_focal = min_focal + F.softplus(raw_focal)
        final_pred = project(points, euler, trans, final_focal)
        final_residual = final_pred - obs
        final_loss = (final_residual.square().sum(dim=-1)[vis]).mean()
        print(f"Final full reprojection MSE: {float(final_loss.cpu()):.4f}")
        print(f"Final focal length: {float(final_focal.cpu()):.4f}")

    points_np = points.detach().cpu().numpy()
    np.save(args.out_dir / "points3d.npy", points_np)
    np.savez(
        args.out_dir / "cameras.npz",
        euler=euler.detach().cpu().numpy(),
        translation=trans.detach().cpu().numpy(),
        focal=np.array([float(final_focal.detach().cpu())], dtype=np.float32),
        view_names=np.array(keys),
    )
    save_obj(args.out_dir / "reconstruction.obj", points_np, args.colors)
    save_loss_csv(args.out_dir / "loss.csv", history)
    save_loss_svg(args.out_dir / "loss.svg", history)
    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
