import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """
    h, w = image.shape[:2]
    dim = image.shape[2] if len(image.shape) > 2 else 1
    
    warped_image = np.full((h, w, dim), 255, dtype=image.dtype)
    
    if len(source_pts) == 0 or len(target_pts) == 0:
        return image
    
    source_pts = source_pts.astype(np.float64)
    target_pts = target_pts.astype(np.float64)
    n = source_pts.shape[0]
    
    # Vectorized distance computation
    diff = source_pts[:, np.newaxis, :] - source_pts[np.newaxis, :, :]  # (n, n, 2)
    R = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Compute RBF parameters D
    D = np.maximum(2 * np.sum((source_pts - target_pts) ** 2, axis=1), 1.0)
    
    # Build RBF matrix K using vectorized phi
    K = 1.0 / (R ** 2 + D[np.newaxis, :])
    
    # Solve for RBF coefficients
    bx = target_pts[:, 0] - source_pts[:, 0]
    by = target_pts[:, 1] - source_pts[:, 1]
    
    try:
        ax = np.linalg.solve(K, bx)
        ay = np.linalg.solve(K, by)
    except np.linalg.LinAlgError:
        ax, _ = np.linalg.lstsq(K, bx, rcond=None)
        ay, _ = np.linalg.lstsq(K, by, rcond=None)
    
    # Vectorized pixel mapping
    # Create mesh grid
    xg, yg = np.meshgrid(np.arange(w), np.arange(h))
    
    # Compute distances from all pixels to all control points
    # Shape: (h, w, n)
    dx = source_pts[:, 0] - xg[:, :, np.newaxis]
    dy = source_pts[:, 1] - yg[:, :, np.newaxis]
    Rp = np.sqrt(dx ** 2 + dy ** 2)
    
    # Compute RBF values for all pixels: (h, w, n)
    Phi = 1.0 / (Rp ** 2 + D[np.newaxis, np.newaxis, :])
    
    # Compute displacements for all pixels: (h, w)
    dx_disp = np.dot(Phi, ax)
    dy_disp = np.dot(Phi, ay)
    
    # Compute new positions
    xnew = np.round(xg + dx_disp).astype(np.int32)
    ynew = np.round(yg + dy_disp).astype(np.int32)
    
    # Accumulation with boundary check
    acc = np.zeros((h, w, dim), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float64)
    
    # Vectorized boundary check
    valid = (xnew >= 0) & (xnew < w) & (ynew >= 0) & (ynew < h)
    
    # Get valid indices
    valid_yp, valid_xp = np.where(valid)
    valid_xnew = xnew[valid_yp, valid_xp]
    valid_ynew = ynew[valid_yp, valid_xp]
    
    # Accumulate
    for i in range(len(valid_yp)):
        yp, xp = valid_yp[i], valid_xp[i]
        xn, yn = valid_xnew[i], valid_ynew[i]
        if dim == 1:
            acc[yn, xn, 0] += image[yp, xp, 0]
        else:
            acc[yn, xn, :] += image[yp, xp, :]
        cnt[yn, xn] += 1
    
    # Average accumulated values
    mapped_mask = cnt > 0
    for c in range(dim):
        warped_image[mapped_mask, c] = np.round(acc[mapped_mask, c] / cnt[mapped_mask]).astype(image.dtype)
    
    # Fill holes using nearest neighbor
    hole_mask = ~mapped_mask
    if np.any(hole_mask) and np.any(mapped_mask):
        try:
            from scipy.interpolate import griddata
            Yg_all, Xg_all = np.meshgrid(np.arange(w), np.arange(h))
            known_x = Xg_all[mapped_mask]
            known_y = Yg_all[mapped_mask]
            query_x = Xg_all[hole_mask]
            query_y = Yg_all[hole_mask]
            
            for c in range(dim):
                known_v = warped_image[:, :, c][mapped_mask]
                vq = griddata((known_x, known_y), known_v, (query_x, query_y), method='nearest')
                warped_image[hole_mask, c] = np.round(vq).astype(image.dtype)
        except:
            pass
    
    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
