import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    
    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    
    # Pad the image efficiently using copyMakeBorder
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    h, w = image.shape[:2]
    center_x, center_y = w / 2.0, h / 2.0
    
    # Pre-compute trigonometric values
    theta = np.radians(rotation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Build combined transformation matrix directly
    # Composition: T @ R @ S @ F
    
    flip_scale = -scale if flip_horizontal else scale
    
    # Combined matrix
    m00 = flip_scale * cos_theta
    m01 = -scale * sin_theta
    m10 = flip_scale * sin_theta
    m11 = scale * cos_theta
    
    # Translation adjusted for center and rotation/scale
    tx = translation_x + center_x - m00 * center_x - m01 * center_y
    ty = -translation_y + center_y - m10 * center_x - m11 * center_y
    
    # Create 2x3 matrix directly for cv2.warpAffine
    M = np.array([[m00, m01, tx],
                  [m10, m11, ty]], dtype=np.float32)
    
    # Apply transformation using OpenCV
    transformed_image = cv2.warpAffine(image, M, (w, h), 
                                       borderValue=(255, 255, 255),
                                       flags=cv2.INTER_LINEAR)
    
    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
