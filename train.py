from os import path, mkdir
from gc import collect
from time import time
from datetime import timedelta
from psutil import Process

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache, memory_allocated
from torch import manual_seed, optim, no_grad, save

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from matplotlib.pyplot import plot, figure, savefig, grid, legend, title

from get_data_loader import get_data_loader

from make_tokenizer import make_tokenizer, c2t

def free():
    
    """
        Free memory and print memory usage.
    """
    
    collect()
    empty_cache()
    print(f'Memory: {Process().memory_info().rss / (1024 * 1024)} MB')

def plot_losses(
    bad: list,
    good: list,
    train: list,
    plot_title: str,
    output_str: str
):
    
    """
        Plot losses for each split.
        
        Parameters:
        - bad (list): Losses for bad_supp.
        - good (list): Losses for good_supp.
        - train (list): Losses for train.
        - plot_title (str): Title of the plot.
    """
    
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

def train(
    loader_name: str,
    tokenizers: dict[str, AutoTokenizer],
    batch_size: int,
    num_batches: int,
    max_length: int,
    lang_code: str,
    num_workers: int,
    epochs: int,
    model: AutoModelForSeq2SeqLM,
    optimizer: optim.Optimizer,
    device: str,
    dev_num_batches: int,
    ckpt: bool,
    output_str: str,
    do_dev: bool,
    log_freq: int,
    get_tokenized: bool
) -> tuple[list, list]:
    
    """
        Train the model on the specified data loader.
        
        Parameters:
        - loader_name (str): Name of the data loader. One of 'bad_supp', 'good_supp', 'train'.
        - tokenizers (dict[str, AutoTokenizer]): Tokenizers for each language.
        - batch_size: (int): Batch size.
        - num_batches (int): Number of batches to train on.
        - max_length: (int): Maximum length of the input sequences.
        - lang_code (str): Language code. Set to None for all languages.
        - num_workers (int): Number of workers for the data loader.
        - epochs: (int): Number of epochs.
        - model (AutoModelForSeq2SeqLM): Model to train.
        - optimizer (optim.Optimizer): Optimizer.
        - device (str): Device.
        - dev_num_batches (int): Number of batches to evaluate on.
        - ckpt (bool): Whether to save checkpoints.
        - output_str (str): Output string.
        - do_dev (bool): Whether to evaluate on dev. Ignored except for 'train' split.
        - log_freq (int): Frequency of logging in batches.
        - get_tokenized (bool): Whether to return tokenized data.
            
        Returns:
        - tuple[list, list]: Train losses and dev losses.
    """

    print('Loading data...') # retrieve appropriate data loader
    free()
    loader = get_data_loader(
        split=loader_name,
        batch_size=batch_size,
        num_batches=num_batches,
        max_length=max_length,
        lang_code=lang_code,
        shuffle=True,
        num_workers=num_workers,
        use_tgts=True, # ignored
        get_tokenized=get_tokenized
    )
    free()
    print('Data loaded.\n')
    
    train_losses = [] # set up for plotting
    dev_losses = []
    
    for epoch in range(epochs): # epoch loop
        epoch_start = time()
        print(f'Epoch {epoch+1} starting...')
        free()
        model.train() # ensure training mode
        for i, batch in enumerate(loader): # batch loop
            if not get_tokenized:
                es_texts, other_texts, lang_token = batch # unpack batch and lang token and tokenize
                tokenized_batch = tokenizers[lang_token](
                    text=es_texts,
                    text_target=other_texts,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=max_length
                )
            else:
                tokenized_batch = batch
            optimizer.zero_grad() # run training step
            outputs = model(**tokenized_batch.to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            item = loss.item() # log and store loss
            if i % log_freq == log_freq - 1:
                print(f'Batch {i+1}/{len(loader)} complete, loss: {item}')
            train_losses.append(item)
            del item # free memory
            del batch
            del outputs
            del loss
            collect()
            empty_cache()
        optimizer.zero_grad() # free memory and log
        free()
        print(f'Epoch {epoch+1} train complete in {str(timedelta(seconds=time()-epoch_start))}.\n')
        
        if do_dev: # evaluate on dev
            print('Loading dev data...') # retrieve dev data loader
            free()
            dev_loaders = get_data_loader(
                split='dev',
                batch_size=batch_size,
                num_batches=dev_num_batches,
                max_length=max_length,
                lang_code=lang_code,
                shuffle=False, # ignored
                num_workers=num_workers,
                use_tgts=True, # for dev loss
                get_tokenized=get_tokenized # ignored
            )
            free()
            print('Dev data loaded.\n')

            print(f'Epoch {epoch+1} eval starting...')
            free()
            model.eval() # ensure evaluation mode
            with no_grad(): # no gradients for evaluation
                # evaluate on each dev data loader - one for each language
                for dev_loader in dev_loaders: # dev loop
                    lang_token = dev_loader.dataset.lang_token # fetch lang token
                    for i, batch in enumerate(dev_loader): # dev batch loop
                        outputs = model(**batch.to(device)) # pretokenized batches
                        loss = outputs.loss
                        item = loss.item() # log and store loss
                        if i % log_freq == log_freq - 1:
                            msg = f'Dev batch {i+1}/{len(dev_loader)} complete'
                            msg += f' (lang={lang_token}), loss: {item}'
                            print(msg)
                        dev_losses.append(item)
                        del item # free memory
                        del batch
                        del outputs
                        del loss
                        collect()
                        empty_cache()
            del dev_loaders # free memory and log 
            free()
            print(f'Epoch {epoch+1} eval complete.\n')

        if ckpt: # save checkpoint
            print('Saving checkpoint...')
            free()
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            ckpts_dir = path.join('outputs', 'ckpts')
            if not path.exists(ckpts_dir):
                mkdir(ckpts_dir)
            save(
                checkpoint,
                path.join(
                    ckpts_dir,
                    f'checkpoint{epoch+1}_{loader_name}_{output_str}.pth'
                )
            )
            del checkpoint
            free()
            print('Done.\n')
            
    del loader # free memory
    free()
        
    return train_losses, dev_losses # return losses for plotting

def main():
    
    """ HYPERPARAMETERS """ # TODO: search for optimal hyperparameters
    overfit           = False                             # overfit on small data to test functionality
    log_freq          = 100     if not overfit else 1     # frequency of logging in batches
    num_workers       = 1                                 # number of workers for data loader
    get_tokenized     = True                              # whether to get tokenized data
    
    batch_size        = 16      if not overfit else 2     # batch size
    max_length        = 384     if not overfit else 16    # maximum length of input sequences
    lang_code         = None    if not overfit else None  # None for all languages
    
    lr                = 1e-5                              # learning rate
    weight_decay      = 1e-2                              # weight decay
    
    bad_epochs        = 1       if not overfit else 0     # num epochs through bad_supp
    do_bad            = True    if not overfit else True  # whether to train on bad_supp
    
    good_epochs       = 3       if not overfit else 0     # num epochs through good_supp
    do_good           = True    if not overfit else True  # whether to train on good_supp
    
    train_epochs      = 10      if not overfit else 10    # every training example is guaranteed included:
    
    dev_num_batches   = None    if not overfit else 20    # None for full dev set
    do_dev            = True    if not overfit else True  # whether to evaluate on dev (ignored for supp data)
    ckpt              = True    if not overfit else False # whether to save checkpoints
    
    bad_num_batches   = int(100_000 / batch_size) if not overfit else 1  # random sampling is used
    good_num_batches  = int(100_000 / batch_size) if not overfit else 1  # random sampling is used
    train_num_batches = int(300_000 / batch_size) if not overfit else 20
    # random sampling for train_num_batches IF train_num_batches * batch_size >= 210368
    
    start = time()
    freeze_support() # parallelism for Windows
    #device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42) # set random seed for reproducibility
    
    # TODO: remove
    print(f'Using {num_workers} workers.')
    
    print('\nLoading model...')
    tokenizers = dict.fromkeys(c2t.values())
    for lang_token in tokenizers: # load tokenizers for each language
        tokenizers[lang_token] = make_tokenizer(lang_token, 'spa_Latn', max_length)
    model_name = 'facebook/nllb-200-distilled-600M'
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size += 8 # 8 new special tokens for languages
    config.max_position_embeddings = max_length # max length built-in to model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    print('Model loaded.')
    model.to(device)
    print(f'Model size on GPU: {memory_allocated(device=device) / 1024**3:.2f} GB.\n')
    
    # output string for saving plots and checkpoints
    output_str = f'{batch_size}_{bad_epochs}_{bad_num_batches}_{good_epochs}'
    output_str += f'_{good_num_batches}_{train_epochs}_{train_num_batches}_{lr}_{weight_decay}'
    if not path.exists('outputs'):
        mkdir('outputs')
    
    optimizer = optim.AdamW( # AdamW optimizer
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
            tokenizers=tokenizers,
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
            ckpt=ckpt,
            output_str=output_str,
            do_dev=False,
            log_freq=log_freq,
            get_tokenized=get_tokenized
        )
        print('Training on bad supp complete.\n')
    else: # let plotting proceed regardless
        bad_train_losses = []
        bad_dev_losses = []
    
    """ TRAINING - GOOD SUPP """
    if do_good:
        print('Training on good supp...')
        good_train_losses, good_dev_losses = train(
            loader_name='good_supp',
            tokenizers=tokenizers,
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
            ckpt=ckpt,
            output_str=output_str,
            do_dev=False,
            log_freq=log_freq,
            get_tokenized=get_tokenized
        )
        print('Training on good supp complete.\n')
    else: # let plotting proceed regardless
        good_train_losses = []
        good_dev_losses = []
    
    """ TRAINING - TRAIN """
    print('Training on train...')
    train_train_losses, train_dev_losses = train(
        loader_name='train',
        tokenizers=tokenizers,
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
        log_freq=log_freq,
        get_tokenized=get_tokenized
    )
    print('Training on train complete.\n')
    
    # produce one train loss plot and one dev loss plot
    # splits are color-coded
    print('Plotting losses...')
    
    plot_losses(bad_train_losses, good_train_losses, train_train_losses, 'train', output_str)
    plot_losses(bad_dev_losses, good_dev_losses, train_dev_losses, 'dev', output_str)
    print(f'Done in {str(timedelta(seconds=time()-start))}\n')

if __name__ == '__main__':
    main()