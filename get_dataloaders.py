import os

from torch import manual_seed

from get_dataloader import get_data_loader

manual_seed(42)

u_dir = os.path.join(
    'train',
    'unprocessed_train'
)

u_mono_dir = os.path.join(u_dir, 'monolingual')
u_st_dir = os.path.join(u_dir, 'shared_task')
u_opus_dir = os.path.join(u_dir, 'opus')

u_mono_files = [os.path.join(u_mono_dir, f) for f in os.listdir(u_mono_dir)]
u_st_files = [os.path.join(u_st_dir, f) for f in os.listdir(u_st_dir)]
u_opus_files = [os.path.join(u_opus_dir, f) for f in os.listdir(u_opus_dir)]

def mono_dataloaders(batch_size, shuffle, num_workers):
    return [
        get_data_loader(
            file,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            mono=True
        ) for file in u_mono_files
    ]

def opus_dataloaders(batch_size, shuffle, num_workers):
    return [
        get_data_loader(
            file,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            mono=False
        ) for file in u_opus_files
    ]
    
def st_dataloaders(batch_size, shuffle, num_workers):
    return [
        get_data_loader(
            file,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            mono=False
        ) for file in u_st_files
    ]
    
def get_all(batch_size, shuffle, num_workers):
    return (
        mono_dataloaders(batch_size, shuffle, num_workers),
        opus_dataloaders(batch_size, shuffle, num_workers),
        st_dataloaders(batch_size, shuffle, num_workers)
    )