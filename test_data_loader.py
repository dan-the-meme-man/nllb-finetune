from torch.multiprocessing import freeze_support
from no_sample_data_loader import get_data_loader

def main():
    freeze_support()
    
    # split: str,
    # batch_size: int,
    # num_batches: int,
    # max_length: int,
    # lang_code: Union[str, None],
    # shuffle: bool,
    # num_workers: int,
    # use_tgts: bool,
    # get_tokenized: bool

    print()
    
    for i in range(1):
        
        print('attempt', i + 1, '\n')
        
        train_loader = get_data_loader(
            split='good_supp',
            batch_size=1,
            num_batches=None,
            max_length=64,
            lang_code=None,
            shuffle=False,
            num_workers=1,
            use_tgts=True,
            get_tokenized=False
        )

        print('loader:', train_loader, len(train_loader), '\n')
        for j, batch in enumerate(train_loader):
            if j % 5000 == 0:
                print(j, batch[0][0][:10], batch[1][0][:10], batch[2])
                print()
        
        print('\n\n\n')

if __name__ == '__main__':
    main()