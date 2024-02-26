import os
from gc import collect

from torch.multiprocessing import freeze_support
from torch.cuda import is_available
from torch.cuda import empty_cache
from torch import manual_seed
from torch import optim

from transformers import AutoModelForSeq2SeqLM

from make_tokenizer import lang_code_to_lang_token as c2t
from get_dataloaders import get_all
from get_dataloader import get_data_loader

def main():

    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        'facebook/nllb-200-distilled-600M'
    )

    model.to(device)

    """ HYPERPARAMETERS """
    bs = 16
    epochs = 3
    lr = 1e-5
    eps = 1e-8
    betas = (0.9, 0.999)
    weight_decay = 0.01

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=eps,
        betas=betas,
        weight_decay=weight_decay
    )

    # mono_dataloaders, opus_dataloaders, st_dataloaders = get_all(bs, True, 4)

    # mono_dataloaders = [
    #     get_data_loader(
    #         os.path.join('train', 'unprocessed_train', 'monolingual', 'cni_0'),
    #         batch_size=1,
    #         shuffle=True,
    #         num_workers=4,
    #         mono=True
    #     )
    # ]

    st_dataloaders = [
        get_data_loader(
            os.path.join('train', 'unprocessed_train', 'shared_task', 'es_ctp.train'),
            batch_size=1,
            shuffle=True,
            num_workers=4,
            mono=False
        )
    ]

    """ TRAINING - MLM WITH MONOLINGUAL DATA """
    for epoch in range(epochs):
        for dataloader in st_dataloaders:

            tokenizer = dataloader.dataset.tokenizer
            
            for i, batch in enumerate(dataloader):
                
                optimizer.zero_grad()
                
                model_inputs = tokenizer(
                    text = batch[0],
                    text_target = batch[1],
                    return_tensors = 'pt',
                    padding = 'max_length',
                    truncation = True,
                    max_length = 1024
                )
                
                # print('mask token id:', tokenizer.mask_token_id)
                # print('mask token:', tokenizer.mask_token)
                
                # print('tokenized input:', model_inputs)
                # exit()
                
                model_inputs = model_inputs.to(device)
                
                outputs = model(**model_inputs)
                
                loss = outputs.loss
                
                loss.backward()
                
                optimizer.step()
                
                collect()
                
                empty_cache()
                
                print(f'Batch {i} complete, loss: {loss}')
        
        print(f'Epoch {epoch} complete')
    
if __name__ == '__main__':
    freeze_support()
    main()