import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim
from torchvision.transforms import InterpolationMode

from tqdm import tqdm
import yaml
import argparse
import warnings

from dataset import SRDataset
from models import BasicSRModel


def main():
    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    parser = argparse.ArgumentParser(
        description='Arguments for training.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_path = config['Dataset']['train_path']
    test_path = config['Dataset']['test_path']
    if config['Dataset']['interpolation'] == 'bilinear':
        interpolation_mode = InterpolationMode.BILINEAR
    elif config['Dataset']['interpolation'] == 'bicubic':
        interpolation_mode = InterpolationMode.BICUBIC
    elif config['Dataset']['interpolation'] == 'nearest-neighbors':
        interpolation_mode = InterpolationMode.NEAREST
    
    train_dataset = SRDataset(train_path, image_size=config['Dataset']['image_size'], upscale_factor=config['upscale_factor'], jitter_val=config['Dataset']['jitter_val'], interpolation=interpolation_mode)
    test_dataset = SRDataset(test_path, image_size=config['Dataset']['image_size'], upscale_factor=config['upscale_factor'], jitter_val=config['Dataset']['jitter_val'], interpolation=interpolation_mode, mode = 'Test')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['Dataset']['batch_size'],
        shuffle=config['Dataset']['shuffle'],
        num_workers=0 if device == 'cpu' else 2,
        drop_last=config['Dataset']['drop_last'],
        pin_memory=config['Dataset']['pin_memory'],
    )
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=config['Dataset']['batch_size'],
        shuffle=config['Dataset']['shuffle'],
        num_workers=0 if device == 'cpu' else 2,
        drop_last=config['Dataset']['drop_last'],
        pin_memory=config['Dataset']['pin_memory'],
    )

    print(f" * Training dataset contains {len(train_dataset)} image(s).")
    print(f" * Testing dataset contains {len(test_dataset)} image(s).")
    for _, batch in enumerate(train_dataloader, 0):
        lr_image, hr_image = batch
        torchvision.io.write_png(lr_image[0, ...].mul(255).byte(), "lr_image.png")
        torchvision.io.write_png(hr_image[0, ...].mul(255).byte(), "hr_image.png")
        break # we deliberately break after one batch as this is just a test

    # learning_rate [1e-2, 1e-3, 1e-5, 1e-6]
    learning_rate = config['Train']['lr']
    save_iterval = config['Train']['save_iterval']
    number_of_epochs = config['Train']['number_of_epochs']
    save_dir = config['Train']['save_dir']

    model = BasicSRModel(upscale_factor = config['upscale_factor'], layers = config['Model']['layers'], residual=config['Model']['residual']).to(device)
    loss_function = nn.L1Loss().to(device)
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=learning_rate)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters in the network: {:d}'.format(num_params))
    
    # Writer will output to ./runs/ directory by default
    if config['log']:
        writer = SummaryWriter(log_dir = config['Train']['log_dir'] )
    # average loss or use global id
    print("Start training")
    for epoch in tqdm(range(number_of_epochs)):
        cnt = 0
        l1 = 0
        psnr = 0
        SSIM = 0
        min_loss = -1
        for batch_id, batch in enumerate(train_dataloader):
    #         global_id = len(train_dataloader) * epoch + batch_id
            low_res, high_res = batch
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            optimizer.zero_grad()
            high_res_prediction = model(low_res)
            loss = loss_function(high_res_prediction, high_res)
            
            l1 += loss
            psnr += -10*torch.log10(mse(high_res_prediction, high_res))
            SSIM += ssim( high_res_prediction, high_res, data_range=1, size_average=True)
            cnt += 1
            
            loss.backward()
            optimizer.step()
            
        if min_loss == -1 or min_loss > (float(l1) / float(cnt)):
            min_loss = float(l1) / float(cnt)
            torch.save(model.state_dict(), ('{:s}/model'.format(save_dir) + ('_res' if config['Model']['residual'] else '') + '_best_lr_{:.0E}.pth'.format(learning_rate)))
        if config['log']:
            writer.add_scalar('L1/train', float(l1) / float(cnt), epoch)
            writer.add_scalar('PSNR/train', float(psnr) / float(cnt), epoch)
            writer.add_scalar('SSIM/train', float(SSIM) / float(cnt), epoch)
        
        cnt = 0
        l1 = 0
        psnr = 0
        SSIM = 0
        for batch_id, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                low_res, high_res = batch
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                high_res_prediction = model(low_res)
                loss = loss_function(high_res_prediction, high_res)

                l1 += loss
                psnr += -10*torch.log10(mse(high_res_prediction, high_res))
                SSIM += ssim( high_res_prediction, high_res, data_range=1, size_average=True)
                cnt += 1
        if config['log']:
            writer.add_scalar('L1/evaluation', float(l1) / float(cnt), epoch)
            writer.add_scalar('PSNR/evaluation', float(psnr) / float(cnt), epoch)
            writer.add_scalar('SSIM/evaluation', float(SSIM) / float(cnt), epoch)
        if (epoch + 1) % save_iterval == 0 or (epoch + 1) == number_of_epochs:    
            torch.save(model.state_dict(), ('{:s}/model'.format(save_dir) + ('_res' if config['Model']['residual'] else '') + '_iter_{:d}_lr_{:.0E}.pth'.format(epoch + 1, learning_rate)))

if __name__ == "__main__":
    main()