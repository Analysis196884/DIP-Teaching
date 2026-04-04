import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from UN_network import UNetGenerator, Discriminator
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 255]
    image = (image + 1.0) * 127.5
    # Clip to [0, 255] and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    num_images = min(num_images, inputs.size(0))
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(generator, discriminator, dataloader, optimizer_g, optimizer_d, criterion_gan, criterion_l1, device, epoch, num_epochs, lambda_l1=100):
    """
    Train the model for one epoch.
    """
    generator.train()
    discriminator.train()

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_d.zero_grad()
        
        # Generate fake images
        fake_images = generator(image_rgb)
        
        # Real loss
        pred_real = discriminator(image_rgb, image_semantic)
        loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
        
        # Fake loss
        pred_fake = discriminator(image_rgb, fake_images.detach())
        loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        
        # Total discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        optimizer_d.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_g.zero_grad()
        
        # GAN loss: trick the discriminator
        pred_fake_g = discriminator(image_rgb, fake_images)
        loss_g_gan = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))
        
        # L1 loss: match the target image
        loss_g_l1 = criterion_l1(fake_images, image_semantic) * lambda_l1
        
        # Total generator loss
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        optimizer_g.step()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_images, 'train_results', epoch)

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}')

def validate(generator, dataloader, criterion_l1, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.
    """
    generator.eval()
    val_loss = 0.0
    # Use a different batch for visualization each time it's saved
    vis_batch_idx = (epoch // 5) % len(dataloader)

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = generator(image_rgb)

            # Compute the loss (L1 only for validation to keep it simple)
            loss = criterion_l1(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs using a rotating batch index
            if epoch % 5 == 0 and i == vis_batch_idx:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss (L1): {avg_val_loss:.4f}')
    return avg_val_loss

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8)

    # Initialize model, loss function, and optimizer
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_g = StepLR(optimizer_g, step_size=200, gamma=0.2)
    scheduler_d = StepLR(optimizer_d, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 200
    log_file = open("train_log.txt", "w")

    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion_gan, criterion_l1, device, epoch, num_epochs)
        avg_val_loss = validate(generator, val_loader, criterion_l1, device, epoch, num_epochs)

        # Write log
        log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], Val L1 Loss: {avg_val_loss:.4f}\n")
        log_file.flush()

        # Step the scheduler after each epoch
        scheduler_g.step()
        scheduler_d.step()

        # Save model checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')
            # Optional: save discriminator state too
            # torch.save(discriminator.state_dict(), f'checkpoints/pix2pix_discriminator_epoch_{epoch + 1}.pth')

    log_file.close()

if __name__ == '__main__':
    main()
