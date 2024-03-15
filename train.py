from os import path, mkdir
from gc import collect
from psutil import Process

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache, memory_allocated
from torch import manual_seed, optim, no_grad, save

from transformers import AutoConfig, AutoModelForSeq2SeqLM

from matplotlib.pyplot import plot, figure, savefig, grid, legend, title

from get_data_loader import get_data_loader

def free():
    collect()
    empty_cache()
    print(f'Memory: {Process().memory_info().rss / (1024 * 1024)} MB')

def train(
    loader_name,
    batch_size,
    num_batches,
    max_length,
    lang_code,
    num_workers,
    epochs,
    model,
    optimizer,
    device,
    dev_num_batches,
    ckpt,
    output_str,
    do_dev,
    log_freq
):
    print('Loading data...')
    free()
    loader = get_data_loader(
        split=loader_name,
        batch_size=batch_size,
        num_batches=num_batches,
        max_length=max_length,
        lang_code=lang_code,
        shuffle=True,
        num_workers=num_workers,
        use_tgts=True # ignored
    )
    free()
    print('Data loaded.\n')
    
    train_losses = []
    dev_losses = []
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} starting...')
        free()
        model.train()
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(**batch.to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            item = loss.item()
            if i % log_freq == log_freq - 1:
                print(f'Batch {i+1}/{len(loader)} complete, loss: {item}')
            train_losses.append(item)
            del outputs
            del loss
            collect()
            empty_cache()
        optimizer.zero_grad()
        free()
        print(f'Epoch {epoch+1} train complete.\n')
        
        if do_dev:
            print('Loading dev data...')
            free()
            dev_loaders = get_data_loader(
                split='dev',
                batch_size=batch_size,
                num_batches=dev_num_batches,
                max_length=max_length,
                lang_code=lang_code,
                shuffle=False, # ignored
                num_workers=num_workers,
                use_tgts=True # for dev loss
            )
            free()
            print('Dev data loaded.\n')
            
            print(f'Epoch {epoch+1} eval starting...')
            free()
            model.eval()
            with no_grad():
                for dev_loader in dev_loaders:
                    lang_token = dev_loader.dataset.lang_token
                    for i, batch in enumerate(dev_loader):
                        outputs = model(**batch.to(device))
                        loss = outputs.loss
                        item = loss.item()
                        if i % log_freq == log_freq - 1:
                            msg = f'Dev batch {i+1}/{len(dev_loader)} complete'
                            msg += f' (lang={lang_token}), loss: {item}'
                            print(msg)
                        dev_losses.append(item)
                        del outputs
                        del loss
                        collect()
                        empty_cache()            
            del dev_loaders
            free()
            print(f'Epoch {epoch+1} eval complete.\n')

        if ckpt:
            print('Saving checkpoint...')
            free()
            checkpoint = {
                'model_state_dict': model.state_dict()
            }
            ckpts_dir = path.join('outputs', 'ckpts')
            if not path.exists(ckpts_dir):
                mkdir(ckpts_dir)
            save(checkpoint, path.join(ckpts_dir, f'checkpoint{epoch+1}_{output_str}.pth'))
            del checkpoint
            free()
            print('Done.\n')
            
    del loader
    free()
        
    return train_losses, dev_losses

def main():
    
    """ HYPERPARAMETERS """
    overfit           = True # TODO: search for optimal hyperparameters
    log_freq          = 100   if not overfit else 1
    num_workers       = 1
    
    batch_size        = 4     if not overfit else 1
    max_length        = 512   if not overfit else 16
    lang_code         = None  if not overfit else 'aym' # None for all languages
    lr                = 1e-5
    weight_decay      = 1e-2
    
    bad_epochs        = 1     if not overfit else 1
    bad_num_batches   = 10000 if not overfit else 1
    do_bad            = True  if not overfit else True
    
    good_epochs       = 3     if not overfit else 1
    good_num_batches  = 10000 if not overfit else 1
    do_good           = True  if not overfit else True
    
    train_epochs      = 10    if not overfit else 30
    train_num_batches = 10000 if not overfit else 5
    
    dev_num_batches   = None  if not overfit else 1     # None for full dev set
    do_dev            = True  if not overfit else True
    ckpt              = True  if not overfit else True
    
    freeze_support()
    #device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42)
    model_name = 'facebook/nllb-200-distilled-600M'
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size += 8 # 8 new special tokens for languages
    print('\nLoading model...')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    print('Model loaded.')
    model.to(device)
    print(f'Model size on GPU: {memory_allocated(device=device) / 1024**3:.2f} GB.\n')
    
    output_str = f'{batch_size}_{bad_epochs}_{bad_num_batches}_{good_epochs}'
    output_str += f'_{good_num_batches}_{train_epochs}_{train_num_batches}_{lr}_{weight_decay}'
    if not path.exists('outputs'):
        mkdir('outputs')
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    
    """ TRAINING - BAD SUPP """
    if do_bad:
        print('Training on bad supp...')
        bad_train_losses, bad_dev_losses = train(
            loader_name='bad_supp',
            batch_size=batch_size,
            num_batches=bad_num_batches,
            max_length=max_length,
            lang_code=lang_code,
            num_workers=num_workers,
            epochs=bad_epochs,
            model=model,
            optimizer=optimizer,
            device=device,
            dev_num_batches=dev_num_batches,
            ckpt=False,
            output_str=output_str,
            do_dev=False,
            log_freq=log_freq
        )
        print('Training on bad supp complete.\n')
    else:
        bad_train_losses = []
        bad_dev_losses = []
    
    """ TRAINING - GOOD SUPP """
    if do_good:
        print('Training on good supp...')
        good_train_losses, good_dev_losses = train(
            loader_name='good_supp',
            batch_size=batch_size,
            num_batches=good_num_batches,
            max_length=max_length,
            lang_code=lang_code,
            num_workers=num_workers,
            epochs=good_epochs,
            model=model,
            optimizer=optimizer,
            device=device,
            dev_num_batches=dev_num_batches,
            ckpt=False,
            output_str=output_str,
            do_dev=False,
            log_freq=log_freq
        )
        print('Training on good supp complete.\n')
    else:
        good_train_losses = []
        good_dev_losses = []
    
    """ TRAINING - TRAIN """
    print('Training on train...')
    train_train_losses, train_dev_losses = train(
        loader_name='train',
        batch_size=batch_size,
        num_batches=train_num_batches,
        max_length=max_length,
        lang_code=lang_code,
        num_workers=num_workers,
        epochs=train_epochs,
        model=model,
        optimizer=optimizer,
        device=device,
        dev_num_batches=dev_num_batches,
        ckpt=ckpt,
        output_str=output_str,
        do_dev=do_dev,
        log_freq=log_freq
    )
    print('Training on train complete.\n')
    
    print('Plotting losses...')
    def plot_losses(bad, good, train, plot_title):
        figure()
        plot(
            range(len(bad)),
            bad,
            label='bad_supp'
        )
        plot(
            range(len(bad), len(bad) + len(good)),
            good,
            label='good_supp'
        )
        plot(
            range(len(bad) + len(good), len(bad) + len(good) + len(train)),
            train,
            label='train'
        )
        title(plot_title)
        grid()
        legend()
        plots_dir = path.join('outputs', 'plots')
        if not path.exists(plots_dir):
            mkdir(plots_dir)
        savefig(path.join(plots_dir, f'{plot_title}_{output_str}.png'))
    plot_losses(bad_train_losses, good_train_losses, train_train_losses, 'train')
    plot_losses(bad_dev_losses, good_dev_losses, train_dev_losses, 'dev')
    print('Done.\n')

if __name__ == '__main__':
    main()