
import argparse
import pickle
import torch
from pathlib import Path
from tqdm.autonotebook import tqdm

import torchkge.models
import torchkge.utils.datasets as torchkge_ds
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from utils import Config, CheckpointManager, SummaryManager
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Training knowledge graph using development knowledge base')
parser.add_argument('--data_dir', default='data', help='Directory containing data')
parser.add_argument('--model_dir', default='experiment', help='Directory to save the expriment result including model')
parser.add_argument('--kg', default='wikidatasets')
parser.add_argument('--model', default='TransR')

parser_for_kg_wiki = parser.add_argument_group(title='wiki')
parser_for_kg_wiki.add_argument('--which', default='companies')
parser_for_kg_wiki.add_argument('--limit', default=0)

parser_for_training = parser.add_argument_group(title='Training')
parser_for_training.add_argument('--epochs', default=100, type=int, help='Epochs for training')
parser_for_training.add_argument('--batch_size', default=256, help='Batch size for training')
parser_for_training.add_argument('--learning_rate', default=0.0004, type=float, help='Learning rate for training')
parser_for_training.add_argument('--ent_dim', default=20, type=int, help='Embedding dimension for Entity')
parser_for_training.add_argument('--rel_dim', default=20, type=int, help='Embedding dimension for Relation')
parser_for_training.add_argument('--margin', default=0.5, type=float, help='Margin for margin ranking loss')
parser_for_training.add_argument('--summary_step', default=50, type=int, help='Summary step for training')

if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    save_dir = model_dir / args.kg /args.which / args.model
    
    assert args.kg in ['wikidatasets', 'fb15k'], "Invalid knowledge graph dataset"
    if args.kg == 'wikidatasets':
        kg_train, kg_valid, _ = torchkge_ds.load_wikidatasets(which=args.which,
                                                                    limit_=args.limit, 
                                                                    data_home=args.data_dir)
    elif args.kg == 'fb15k':
        kg_train, kg_valid, _ = torchkge_ds.load_fb15k(data_home=args.data_dir)
    
    
    assert args.model in ['TransE', 'TransR', 'DistMult'], "Invalid Knowledge Graph Embedding Model"
    if args.model == 'TransE':
        model = torchkge.models.TransEModel(args.ent_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    elif args.model == 'DistMult':
        model = torchkge.models.DistMultModel(args.ent_dim, kg_train.n_ent, kg_train.n_rel)
    elif args.model == 'TransR':
        model = torchkge.models.TransRModel(args.ent_dim, args.rel_dim, kg_train.n_ent, kg_train.n_rel)
    
    criterion = MarginLoss(args.margin)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()
        criterion.cuda()
    
    writer = SummaryWriter(save_dir / f'runs_{args.model}')
    checkpoint_manager = CheckpointManager(save_dir)
    summary_manager = SummaryManager(save_dir)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    sampler = BernoulliNegativeSampler(kg_train)
    tr_dl = DataLoader(kg_train, batch_size = args.batch_size, use_cuda='all')
    val_dl = DataLoader(kg_train, batch_size = args.batch_size, use_cuda='all')
    
    best_val_loss = 1e+10
    
    for epoch in tqdm(range(args.epochs), desc='epochs'):
        
        tr_loss = 0
        model.train()
        
        for step, batch in enumerate(tr_dl):
            h, t, r = batch[0], batch[1], batch[2]
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
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)
            with torch.no_grad():
                pos, neg = model(h, t, n_h, n_t, r)
                loss = criterion(pos, neg)
                val_loss += loss.item()
        val_loss /= (step+1)
        
        
        if (epoch+1) % args.summary_step == 0:
            tqdm.write('Epoch {} | train loss: {:.5f}, valid loss: {:.5f}'.format(epoch+1, tr_loss, val_loss))
        model.normalize_parameters()
        is_best = val_loss < best_val_loss
        if is_best:
            state = {'epoch': epoch, 
                     'model_state_dict': model.state_dict(), 
                     'optimizer': optimizer.state_dict()}
            summary = {'training loss': tr_loss, 'validation loss': val_loss}
            
            summary_manager.update(summary)
            summary_manager.save(f'summary_{args.model}.json')
            checkpoint_manager.save_checkpoint(state, f'best_{args.model}.tar')
            best_val_loss = val_loss
