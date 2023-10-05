import torch
from torch.optim import Adam
from backward_diffusion import model
from forward_diffusion import train_loader
from utils import sample_plot_image, get_loss

BATCH_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 15

for epoch in range(epochs):
    print(f"Running epoch: {epoch}")
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # print(batch.shape)
        # print(batch[0].shape)
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch, t)
        # loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Training Loss: {loss.item()} ")
            sample_plot_image()