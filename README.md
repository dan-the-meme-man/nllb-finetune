# nllb-finetune

Note to Kenton: place the data we shared in the same directory as all the files in this repo, and then just run the `train.py` file.

You should also be able to just run `conda install -r requirements.txt`, but I don't regularly use `conda`. If it should be named `pytorch` instead of `torch`, feel free to fix that.

Unfortunately, the model is too big to even run reasonably on my laptop, so I can't test whether the model learns anything, even on a very small set.

IMPORTANT: at present, the script is configured to train only on 10 examples of one dataset with batch size 1. I would like to confirm whether it will learn those 10 examples before I move on to trying a larger dataset.
