from os import path, mkdir
from gc import collect
from psutil import Process

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache, memory_allocated
from torch import manual_seed, optim, no_grad, save, tensor, Tensor
from torch.nn import Module

from matplotlib.pyplot import plot, figure, savefig, grid, legend, title

import sentencepiece as spm

from get_data_loader import get_data_loader

from make_tokenizer import t2c

from mamba_model import MambaModel

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

def tokenize_batch(
    tokenizer: spm.SentencePieceProcessor,
    batch: tuple[list[str], list[str], str]
) -> Tensor:
        
        """
            Tokenize a batch of text.
            
            Parameters:
            - tokenizer (spm.SentencePieceProcessor): Tokenizer.
            - batch (list[str]): Batch of text.
                
            Returns:
            - torch.Tensor: Tokenized batch.
        """
        
        es_texts, other_texts, lang_token = batch # unpack batch and lang token
        
        assert len(es_texts) == len(other_texts) # ensure equal lengths
        
        tokenized_batch = []
        for es_text, other_text in zip(es_texts, other_texts):
            tokenized_batch.append(
                tokenizer.EncodeAsIds(es_text) +
                [tokenizer.PieceToId('<' + t2c[lang_token] + '>')] +
                tokenizer.EncodeAsIds(other_text)
            )
            
        # tokenized as if: hola que tal <aym> (same sentence in Aymara)
        # TODO: this is suitable for next token prediction, but maybe not correct for seq2seq
        # it could be fine, we'll need to have a generate method to test
        
        return tensor(tokenized_batch)

def train(
    loader_name: str,
    tokenizer: spm.SentencePieceProcessor,
    batch_size: int,
    num_batches: int,
    max_length: int,
    lang_code: str,
    num_workers: int,
    epochs: int,
    model: Module,
    optimizer: optim.Optimizer,
    device: str,
    dev_num_batches: int,
    ckpt: bool,
    output_str: str,
    do_dev: bool,
    log_freq: int
):
    
    """
        Train the model on the specified data loader.
        
        Parameters:
        - loader_name (str): Name of the data loader. One of 'bad_supp', 'good_supp', 'train'.
        - tokenizers (dict[str, AutoTokenizer]): Tokenizers for each language.
        - batch_size (int): Batch size.
        - num_batches (int): Number of batches to train on.
        - max_length (int): Maximum length of the input sequences.
        - lang_code (str): Language code. Set to None for all languages.
        - num_workers (int): Number of workers for the data loader.
        - epochs (int): Number of epochs.
        - model (AutoModelForSeq2SeqLM): Model to train.
        - optimizer (optim.Optimizer): Optimizer.
        - device (str): Device.
        - dev_num_batches (int): Number of batches to evaluate on.
        - ckpt (bool): Whether to save checkpoints.
        - output_str (str): Output string.
        - do_dev (bool): Whether to evaluate on dev. Ignored except for 'train' split.
        - log_freq (int): Frequency of logging in batches.
            
        Returns:
        - list: Train losses.
        - list: Dev losses.
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
        use_tgts=True # ignored
    )
    free()
    print('Data loaded.\n')
    
    train_losses = [] # set up for plotting
    dev_losses = []
    
    for epoch in range(epochs): # epoch loop
        print(f'Epoch {epoch+1} starting...')
        free()
        model.train() # ensure training mode
        for i, batch in enumerate(loader): # batch loop
            tokenized_batch = tokenize_batch(tokenizer, batch)
            optimizer.zero_grad() # run training step
            outputs = model(tokenized_batch.to(device))
            loss = None # TODO: choose loss function and compute here
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
        print(f'Epoch {epoch+1} train complete.\n')
        
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
                use_tgts=True # for dev loss
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
                        outputs = model(tokenized_batch.to(device)) # pretokenized batches
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
                'model_state_dict': model.state_dict()
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
    overfit           = True                             # overfit on small data to test functionality
    log_freq          = 100     if not overfit else 1     # frequency of logging in batches
    num_workers       = 1                                 # number of workers for data loader
    
    batch_size        = 4       if not overfit else 2     # batch size
    max_length        = 384     if not overfit else 16    # maximum length of input sequences
    lang_code         = None    if not overfit else None  # None for all languages
    
    lr                = 1e-5                              # learning rate
    weight_decay      = 1e-2                              # weight decay
    
    tokenizer_type    = 'bpe'                             # sentencepiece model type
    tokenizer_type    = 'unigram'
    
    layers            = 6       if not overfit else 2     # number of layers in Mamba
    d_model           = 768     if not overfit else 16    # dimension of model
    d_state           = 256     if not overfit else 16    # dimension of state
    d_conv            = 4       if not overfit else 4     # dimension of convolution
    expand            = 2       if not overfit else 2     # expansion factor
    
    bad_epochs        = 1       if not overfit else 0     # num epochs through bad_supp
    bad_num_batches   = 25_000  if not overfit else 1     # random sampling is used
    do_bad            = True    if not overfit else True  # whether to train on bad_supp
    
    good_epochs       = 3       if not overfit else 0     # num epochs through good_supp
    good_num_batches  = 25_000  if not overfit else 1     # random sampling is used
    do_good           = True    if not overfit else True  # whether to train on good_supp
    
    train_epochs      = 10      if not overfit else 10    # every training example is guaranteed included:
    train_num_batches = 75_000  if not overfit else 20    # IF train_num_batches * batch_size >= 210368
    
    dev_num_batches   = None    if not overfit else 20    # None for full dev set
    do_dev            = True    if not overfit else True  # whether to evaluate on dev (ignored for supp data)
    ckpt              = True    if not overfit else False # whether to save checkpoints
    
    freeze_support() # parallelism for Windows
    #device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42) # set random seed for reproducibility
    
    print('\nLoading model...')
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(path.join('vocab', tokenizer_type + '.model'))
    model = MambaModel(
        layers=layers,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        max_length=max_length
    )
    print('Model loaded.')
    model.to(device)
    print(f'Model size on GPU: {memory_allocated(device=device) / 1024**3:.2f} GB.\n')
    
    # output string for saving plots and checkpoints
    output_str = f'{batch_size}_{bad_epochs}_{bad_num_batches}_{good_epochs}'
    output_str += f'_{good_num_batches}_{train_epochs}_{train_num_batches}_{lr}_{weight_decay}'
    output_str += f'_{tokenizer_type}_{layers}_{d_model}_{d_state}_{d_conv}_{expand}'
    if not path.exists('outputs'):
        mkdir('outputs')
    
    optimizer = optim.AdamW( # AdamW optimizer
        model.parameters(), # TODO: try different optimizers?
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
            tokenizer=tokenizer,
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
            log_freq=log_freq
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
            tokenizer=tokenizer,
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
            log_freq=log_freq
        )
        print('Training on good supp complete.\n')
    else: # let plotting proceed regardless
        good_train_losses = []
        good_dev_losses = []
    
    """ TRAINING - TRAIN """
    print('Training on train...')
    train_train_losses, train_dev_losses = train(
        loader_name='train',
        tokenizer=tokenizer,
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
    
    # produce one train loss plot and one dev loss plot
    # splits are color-coded
    print('Plotting losses...')
    
    plot_losses(bad_train_losses, good_train_losses, train_train_losses, 'train')
    plot_losses(bad_dev_losses, good_dev_losses, train_dev_losses, 'dev')
    print('Done.\n')

if __name__ == '__main__':
    main()