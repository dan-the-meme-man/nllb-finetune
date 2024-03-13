import os
from gc import collect

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache
from torch import manual_seed, optim, no_grad, save

from transformers import AutoConfig, AutoModelForSeq2SeqLM

import matplotlib.pyplot as plt

from get_data_loader import get_data_loader

def train(
    loader_name,
    batch_size,
    num_batches,
    num_workers,
    epochs,
    model,
    optimizer,
    device,
    dev_loaders,
    ckpt,
    output_str
):
    
    print('Loading data...')
    loader = get_data_loader(
        split=loader_name,
        batch_size=batch_size,
        num_batches=num_batches,
        shuffle=True,
        num_workers=num_workers
    )
    print('Data loaded.\n')
    
    train_losses = []
    dev_losses = []
    
    for epoch in range(epochs):
        
        print(f'Epoch {epoch+1} starting...')
        model.train()
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(**batch.to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            collect()
            empty_cache()
            item = loss.item()
            print(f'Batch {i}/{len(loader)} complete, loss: {item}')
            train_losses.append(item)
        print(f'Epoch {epoch+1} train complete.\n')
        
        # print(f'Epoch {epoch+1} eval starting...')
        # model.eval()
        # with no_grad():
        #     for loader in dev_loaders:
        #         for i, batch in enumerate(loader):
        #             outputs = model(**batch.to(device))
        #             loss = outputs.loss
        #             item = loss.item()
        #             print(f'Dev batch {i}/{len(loader)} complete, loss: {item}')
        #             dev_losses.append(item)    
        # print(f'Epoch {epoch+1} eval complete.\n')

        if ckpt:
            print('Saving checkpoint...')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            ckpts_dir = os.path.join('outputs', 'ckpts')
            if not os.path.exists(ckpts_dir):
                os.mkdir(ckpts_dir)
            save(checkpoint, os.path.join(ckpts_dir, f'checkpoint_{epoch+1}_{output_str}.pth'))
            print('Done.\n')
        
    return train_losses, dev_losses

def main():
    
    freeze_support()
    
    #device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42)

    model_name = 'facebook/nllb-200-distilled-600M'
    
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size += 8 # 8 new special tokens for languages
    
    print('Loading model...')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    print('Model loaded.\n')
    
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.size())
    #     print('\n')
    # exit()

    model.to(device)

    """ HYPERPARAMETERS """
    overfit = True # TODO: search for optimal hyperparameters
    batch_size        = 8     if not overfit else 1
    bad_epochs        = 1     if not overfit else 1
    bad_num_batches   = 10000 if not overfit else 10
    good_epochs       = 3     if not overfit else 1
    good_num_batches  = 10000 if not overfit else 10
    train_epochs      = 10    if not overfit else 1
    train_num_batches = 10000 if not overfit else 10
    lr = 1e-5
    weight_decay = 0.01
    
    output_str = f'{batch_size}_{bad_epochs}_{bad_num_batches}_{good_epochs}'
    output_str += f'_{good_num_batches}_{train_epochs}_{train_num_batches}_{lr}_{weight_decay}'
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    
    print('Loading dev data...')
    num_workers = 2
    dev_loaders = get_data_loader(
        split='dev',
        batch_size=1, # ignored
        num_batches=-1, # ignored
        shuffle=False, # ignored
        num_workers=num_workers
    )
    print('Dev data loaded.\n')
    
    """ TRAINING - BAD SUPP """
    print('Training on bad supp...')
    bad_train_losses, bad_dev_losses = train(
        loader_name='bad_supp',
        batch_size=batch_size,
        num_batches=bad_num_batches,
        num_workers=num_workers,
        epochs=bad_epochs,
        model=model,
        optimizer=optimizer,
        device=device,
        dev_loaders=dev_loaders,
        ckpt=False,
        output_str=output_str
    )
    print('Training on bad supp complete.\n')
    
    """ TRAINING - GOOD SUPP """
    print('Training on good supp...')
    good_train_losses, good_dev_losses = train(
        loader_name='bad_supp',
        batch_size=batch_size,
        num_batches=good_num_batches,
        num_workers=num_workers,
        epochs=good_epochs,
        model=model,
        optimizer=optimizer,
        device=device,
        dev_loaders=dev_loaders,
        ckpt=False,
        output_str=output_str
    )
    print('Training on good supp complete.\n')
    
    """ TRAINING - TRAIN """
    print('Training on train...')
    train_train_losses, train_dev_losses = train(
        loader_name='bad_supp',
        batch_size=batch_size,
        num_batches=train_num_batches,
        num_workers=num_workers,
        epochs=train_epochs,
        model=model,
        optimizer=optimizer,
        device=device,
        dev_loaders=dev_loaders,
        ckpt=True,
        output_str=output_str
    )
    print('Training on train complete.\n')
    
    print('Plotting losses...')
    def plot_losses(bad, good, train, title):
        plt.figure()
        plt.plot(
            range(len(bad)),
            bad,
            label='bad_supp'
        )
        plt.plot(
            range(len(bad), len(bad) + len(good)),
            good,
            label='good_supp'
        )
        plt.plot(
            range(len(bad) + len(good), len(bad) + len(good) + len(train)),
            train,
            label='train'
        )
        plt.title(title)
        plt.grid()
        plt.legend()
        plots_dir = os.path.join('outputs', 'plots')
        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)
        plt.savefig(os.path.join(plots_dir, f'{title}_{output_str}.png'))
        
    plot_losses(bad_train_losses, good_train_losses, train_train_losses, 'train')
    plot_losses(bad_dev_losses, good_dev_losses, train_dev_losses, 'dev')
    print('Done.\n')
    
if __name__ == '__main__':
    main()