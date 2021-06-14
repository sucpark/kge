import pickle
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

import torchkge.models
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from utils import CheckpointManager, SummaryManager, DataParallel
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Training knowledge graph using development knowledge base')
parser.add_argument('--data_dir', default='data', help='Directory containing data')
parser.add_argument('--save_dir', default='experiment', help='Directory to save the experiment results')
parser.add_argument('--data', default='wikidatasets')
parser.add_argument('--model', default='TransE')

parser_for_kg_wiki = parser.add_argument_group(title='wiki')
parser_for_kg_wiki.add_argument('--which', default='companies')
parser_for_kg_wiki.add_argument('--limit', default=0, type=int)

parser_for_training = parser.add_argument_group(title='Training')
parser_for_training.add_argument('--epochs', default=10, type=int, help='Epochs for training')
parser_for_training.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser_for_training.add_argument('--learning_rate', default=0.0004, type=float, help='Learning rate for training')
parser_for_training.add_argument('--ent_dim', default=20, type=int, help='Embedding dimension for Entity')
parser_for_training.add_argument('--rel_dim', default=20, type=int, help='Embedding dimension for Relation')
parser_for_training.add_argument('--margin', default=0.5, type=float, help='Margin for margin ranking loss')
parser_for_training.add_argument('--summary_step', default=2, type=int, help='Summary step for training')
parser_for_training.add_argument('--gpu', default=None, type=list, help='Set GPU for training')

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir) / args.data
    save_dir = Path(args.save_dir) / args.data
    
    experiment_summary = {'data': args.data, 'which': args.which, 'limit': args.limit,
                          'model': args.model, '# of epochs': args.epochs, 'batch size': args.batch_size,
                          'learning rate': args.learning_rate, 'margin': args.margin,
                          'entity dimension': args.ent_dim, 'relation dimension': args.rel_dim}
    experiment_summary = dict(**experiment_summary)
    experiment_summary = {'Experiment Summary': experiment_summary}
    
    assert args.data in ['wikidatasets', 'fb15k'], "Invalid knowledge graph dataset"
    if args.data == 'wikidatasets':
        data_dir = data_dir / args.which
        save_dir = save_dir / args.which
    save_dir = save_dir / args.model

    with open(data_dir / 'kg_train.pkl', mode='rb') as io:
        kg_train = pickle.load(io)
    with open(data_dir / 'kg_valid.pkl', mode='rb') as io:
        kg_valid = pickle.load(io)

    assert args.model in ['TransE', 'TransR', 'DistMult', 'TransD'], "Invalid Knowledge Graph Embedding Model"
    if args.model == 'TransE':
        model = torchkge.models.TransEModel(args.ent_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    elif args.model == 'DistMult':
        model = torchkge.models.DistMultModel(args.ent_dim, kg_train.n_ent, kg_train.n_rel)
    elif args.model == 'TransR':
        model = torchkge.models.TransRModel(args.ent_dim, args.rel_dim, kg_train.n_ent, kg_train.n_rel)
    elif args.model == 'TransD':
        model = torchkge.models.TransDModel(args.ent_dim, args.rel_dim, kg_train.n_ent, kg_train.n_rel)
    
    criterion = MarginLoss(args.margin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        print('gpu is available')
        torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        print('multiple gpus are available')
        if args.gpu is not None:
            model = DataParallel(model, device_ids=args.gpu)
        else:
            model = DataParallel(model)
    model.to(device)
    criterion.to(device)

    writer = SummaryWriter(save_dir / f'runs_{args.model}')
    checkpoint_manager = CheckpointManager(save_dir)
    summary_manager = SummaryManager(save_dir)
    summary_manager.update(experiment_summary)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    sampler = BernoulliNegativeSampler(kg_train)
    tr_dl = DataLoader(kg_train, batch_size=args.batch_size)
    val_dl = DataLoader(kg_valid, batch_size=args.batch_size)

    best_val_loss = 1e+10
    for epoch in tqdm(range(args.epochs), desc='epochs'):
        tr_loss = 0
        model.train()
        
        for step, batch in enumerate(tr_dl):
            h, t, r = map(lambda elm: elm.to(device), batch)
            n_h, n_t = sampler.corrupt_batch(h, t, r)
            
            optimizer.zero_grad()
            
            pos, neg = model(h, t, n_h, n_t, r)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= (step+1)
        
        model.eval()
        val_loss = 0
        for step, batch in enumerate(val_dl):
            h, t, r = map(lambda elm: elm.to(device), batch)
            n_h, n_t = sampler.corrupt_batch(h, t, r)
            with torch.no_grad():
                pos, neg = model(h, t, n_h, n_t, r)
                loss = criterion(pos, neg)
                val_loss += loss.item()
        val_loss /= (step+1)
        writer.add_scalars('loss', {'train': tr_loss, 'val': val_loss}, epoch)
        if (epoch+1) % args.summary_step == 0:
            tqdm.write('Epoch {} | train loss: {:.5f}, valid loss: {:.5f}'.format(epoch+1, tr_loss, val_loss))
        model.normalize_parameters()
        is_best = val_loss < best_val_loss
        if is_best:
            state = {'epoch': epoch, 
                     'model_state_dict': model.state_dict(), 
                     'optimizer': optimizer.state_dict()}
            summary = {'training loss': round(tr_loss, 4), 'validation loss': round(val_loss, 4)}
            summary = dict(**summary)
            summary = {'Training Summary': summary}
            
            summary_manager.update(summary)
            summary_manager.save(f'summary_{args.model}.json')
            checkpoint_manager.save_checkpoint(state, f'best_{args.model}.tar')
            best_val_loss = val_loss
