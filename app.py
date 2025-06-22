# app.py (Updated to match the screenshot)

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Model architecture definition (This must be identical to your training script) ---
# (This section is unchanged)
LATENT_DIMS = 10

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear_mean = nn.Linear(128, latent_dims)
        self.linear_log_var = nn.Linear(128, latent_dims)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mean =  self.linear_mean(x)
        log_var = self.linear_log_var(x)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 128)
        self.linear2 = nn.Linear(128, 3 * 3 * 32)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.conv_transpose1 = nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv_transpose2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.batch2 = nn.BatchNorm2d(8)
        self.conv_transpose3 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.unflatten(x)
        x = F.relu(self.batch1(self.conv_transpose1(x)))
        x = F.relu(self.batch2(self.conv_transpose2(x)))
        x = torch.tanh(self.conv_transpose3(x))
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        x_recon = self.decoder(z)
        return x_recon, mean, log_var
# --- End of Model Architecture ---


# Function to load the model (unchanged)
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = VariationalAutoencoder(LATENT_DIMS)
    model.load_state_dict(torch.load('vae_mnist.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# Load the model
vae_model = load_model()


# --- Streamlit User Interface (Updated to match the screenshot) ---

st.title("Handwritten Digit Image Generator")

st.write("Generate synthetic MNIST-like images using your trained model.")

# User input with label from screenshot
selected_digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

# Button with static label from screenshot
if st.button("Generate Images"):
    # Subheader that dynamically updates based on the selection
    st.subheader(f"Generated images of digit {selected_digit}")

    with torch.no_grad():
        # Display 5 images in a row
        cols = st.columns(5)
        for i in range(5):
            # Sample a random latent vector
            z = torch.randn(1, LATENT_DIMS)
            
            # Generate an image
            generated_image = vae_model.decoder(z)
            
            # Post-process for display
            image_np = generated_image.view(28, 28).cpu().numpy()
            image_np = (image_np + 1) / 2.0 
            
            # Display the image in one of the columns with the updated caption style
            cols[i].image(image_np, caption=f"Sample {i+1}", use_column_width=True)