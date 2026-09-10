import yamlargparse, os, random
import numpy as np
import torch

from dataloader.dataloader import dataloader_wrapper
from solver import Solver
from subject_embedder import SubjectEncodings, SubjectEmbeddingType, FiLMNetwork

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    args.device = device
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        print(f"Rank: {args.local_rank}, World size: {args.world_size}, Device: {device}")
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, init_method='env://', world_size=args.world_size)

    from networks import network_wrapper
    model = network_wrapper(args)
    if args.distributed: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if (args.distributed and args.local_rank ==0) or args.distributed == False: 
        print("started on " + args.checkpoint_dir + '\n')
        print(args)
        print(model)
        print("\nTotal number of parameters: {} \n".format(sum(p.numel() for p in model.parameters())))
        print("\nTotal number of trainable parameters: {} \n".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_learning_rate)

    train_sampler, train_generator = dataloader_wrapper(args,'train')
    print("Train sampler: loaded")
    _, val_generator = dataloader_wrapper(args, 'val')
    print("Validation sampler: loaded")
    _, test_generator = dataloader_wrapper(args, 'test')
    print("Test sampler: loaded")
    args.train_sampler=train_sampler


    solver = Solver(args=args,
                model = model,
                optimizer = optimizer,
                train_data = train_generator,
                validation_data = val_generator,
                test_data = test_generator
                ) 
    print("Start training")
    if not args.evaluate_only:
        solver.train()
        
'''
Phase 1:
In this phase, we do contrastive pretraining to build discriminatory ability of eeg encoder and subject embeddings. We provide random noise to the audio encoder and aim to learn a mapping from eeg to the latent mask. We do this by using a one of the subject embedding methods, freezing the trained speech encoder, and performing contrastive learning with the eeg encoding, attended speaker encoding, and unattended speaker encoding. We generate the combined eeg audio embedding which acts as the mask. Then we encode both the attended, unattended, and mixed audios, divide the attended and unattended audio encodings by the mixed audio encoding, producing the attended and unattended audio masks. We then compute the contrastive loss between the eeg mask and the attended and unattended audio masks with the attended mask being the positive sample and the unattended mask being the negative sample. The eeg encoder and subject embeddings are trained to minimize this contrastive loss.
'''
def phase1():
    return

'''
Phase 2:
In this phase, we train to directly predict the latent mask. Similarly to phase 1, we freeze the trained speech encoder and use the eeg encoder and subject embeddings to predict the latent mask. We then encode the attended speech, divide it by the encoded mixed speech, and train the model to minimize cosine similarity between that and the predicted latent mask.
'''
def phase2():
    return

'''
Phase 3:
In this phase, we fine-tune the model for the final task. We unfreeze the trained speech encoder, freeze the subject embeddings, and train the model end-to-end as usual.
'''
def phase3():
    return