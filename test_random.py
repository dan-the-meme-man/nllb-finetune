from get_data_loader import get_data_loader

from torch.multiprocessing import freeze_support

def main():
    
    freeze_support()
    
    loader = get_data_loader(
        split='train',
        batch_size=16,
        num_batches=50,
        max_length=64,
        lang_code=None,
        shuffle=True,
        num_workers=1,
        use_tgts=True, # ignored
        get_tokenized=False
    )

    for batch in loader:
        print(batch)
        break

    loader = get_data_loader(
        split='train',
        batch_size=16,
        num_batches=50,
        max_length=64,
        lang_code=None,
        shuffle=True,
        num_workers=1,
        use_tgts=True, # ignored
        get_tokenized=False
    )

    for batch in loader:
        print(batch)
        break
    
if __name__ == '__main__':
    main()