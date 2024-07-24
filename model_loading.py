import torch
import os
from model import GPTConfig, GPT


def load_model(configs, device):
    print(f"Loading tyhe model from {configs['out_dir']}")
    # model init
    model_args = dict(n_layer=configs['n_layer'], n_head=configs['n_head'], n_embd=configs['n_embd'], block_size=configs['block_size'],
                  bias=configs['bias'], vocab_size=None, dropout=configs['dropout']) # start with model_args from command line
    # resume training from a checkpoint.
    ckpt_path = os.path.join(configs['out_dir'], 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    return model, iter_num, best_val_loss, checkpoint


if __name__ == '__main__':
    print('')