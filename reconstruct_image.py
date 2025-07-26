import os
import time
import numpy as np
import nibabel as nib
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import inversefed
from classificator import *
from helpers import *


start_time = time.time()

# ============================ CHANGE THESE PARAMETERS ============================
epochs = 1

params = {
    "batch_size": 5,
    "imagex": 160,
    "imagey": 192,
    "imagez": 160,
    "column": "Group_bin",
}

image_path = 'reconstructed_images'
os.makedirs(image_path, exist_ok=True)

ground_truth_path = 'reconstructed_images/ground_truth.nii.gz'
reconstructed_path = 'reconstructed_images/reconstructed.nii.gz'

model_path = 'trained_models'
os.makedirs(model_path, exist_ok=True)

trained_model_name = 'test_model.pth'
new_model_name = 'test_model.pth'

data_path = 'data_csv'
csv_dir = 'data_csv'
nifti_dir = '/work/forkert_lab/mitacs_dataset/affine_using_nifty_reg'
num_images = 1
# =================================================================================


setup = inversefed.utils.system_startup()
device, dtype = setup["device"], setup["dtype"]

train_csv = os.path.join(csv_dir, "train_pd_complete_adni.csv")
val_csv = os.path.join(csv_dir, "test_pd_complete_adni.csv")

train_ds = CSVNiftiDataset(train_csv, nifti_dir, params["column"])
val_ds   = CSVNiftiDataset(val_csv,   nifti_dir, params["column"])
train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=1, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

print("[INFO] Computing global mean/std ...")
dm = 0.1495
ds = 0.1982

print(f"[INFO] mean={dm:.4f}, std={ds:.4f}")


model = SFCN().to(device=device, dtype=dtype)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=np.exp(-0.1))

# ============================ TRAINING STARTS ============================
# Comment if you want to skip training and use pretrained model
print("[INFO] Start training ...")
best_val_loss, patience, patience_counter = float("inf"), 10, 0
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    print("Epoch {}: Train Loss={:.4f}, Train Acc={:.4f}, Val Loss={:.4f}, Val Acc={:.4f}".format(epoch+1, train_loss, train_acc, val_loss, val_acc))


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(model_path, new_model_name))
        print("Model saved at epoch {}".format(epoch+1))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
print("[INFO] Model training finished.")
# ============================ TRAINING ENDS ==============================


# Comment if you're training new model
model.load_state_dict(torch.load(os.path.join(model_path, trained_model_name)))

model.eval()
model.zero_grad()


# ============================ UPLOAD GROUND TRUTH IMAGE ============================
nii = nib.load(ground_truth_path)
data = nii.get_fdata()
data = np.asarray(data, dtype=np.float32)
ground_truth = torch.as_tensor(data, **setup)
ground_truth = ground_truth.sub(dm).div(ds)
ground_truth = ground_truth.unsqueeze(0).unsqueeze(0).contiguous()
img_shape = ground_truth.shape[1:]
# =================================================================================


# ============================ RECONSTRUCTION STARTS ==============================
loss_fn = nn.CrossEntropyLoss()
model_output = model(ground_truth)            
labels = torch.tensor([1], dtype=torch.long, device=setup["device"])
target_loss = loss_fn(model_output, labels)

input_gradient = torch.autograd.grad(target_loss, model.parameters())
input_gradient = [grad.detach() for grad in input_gradient]

config = dict(signed=True,
            boxed=True,
            cost_fn='sim',
            indices='def',
            weights='equal',
            lr=0.03,
            optim='adam',
            restarts=1,
            max_iterations=300,
            total_variation=1e-5,
            init='randn',
            filter='gauss5',
            lr_decay=True,
            scoring_choice='loss')

rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=False)
# ============================ RECONSTRUCTION ENDS ================================


# Compute stats
test_mse = (output - ground_truth).pow(2).mean().item()
feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1 / ds)

output_np = output[0].squeeze().cpu().numpy()
nib.save(nib.Nifti1Image(output_np, nii.affine), reconstructed_path)

print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

print(datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
print('---------------------------------------------------')
print(f'Finished computations with time: {str(timedelta(seconds=time.time() - start_time))}')
print('-------------Job finished.-------------------------')
