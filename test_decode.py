from torch.multiprocessing import freeze_support
from torch.cuda import is_available, memory_allocated
from torch import no_grad

from transformers import AutoConfig, AutoModelForSeq2SeqLM

from get_data_loader import get_data_loader

def main():
    
    freeze_support()

    device = 'cuda' if is_available() else 'cpu'

    model_name = 'facebook/nllb-200-distilled-600M'
        
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size += 8 # 8 new special tokens for languages

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )

    model.to(device)
    print(f'Model size on GPU: {memory_allocated(device=device) / 1024**3:.2f} GB')

    max_length = 256

    dev_loaders = get_data_loader(
        split='dev',
        batch_size=2,
        num_batches=5,
        max_length=max_length,
        lang_code='aym',
        shuffle=False, # ignored
        num_workers=1
    )

    with no_grad():
        model.eval()
        for dev_loader in dev_loaders:
            lang_token = dev_loader.dataset.lang_token
            tokenizer = dev_loader.dataset.tokenizer
            for i, batch in enumerate(dev_loader):
                outputs = model.generate(
                    **batch.to(device),
                    max_length=max_length,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                print(f'Lang: {lang_token}, Decoded: {decoded}')
                
if __name__ == '__main__':
    main()