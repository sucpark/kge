import argparse
import torch
from pathlib import Path
from torchkge.utils import MarginLoss, DataLoader
import torchkge.models
import torchkge.utils.datasets as torchkge_ds
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.evaluation import LinkPredictionEvaluator, TripletClassificationEvaluator
from utils import CheckpointManager, SummaryManager

parser = argparse.ArgumentParser(description="Evaluating knowledge graph embedding using development database")
parser.add_argument('--data_dir', default='data', help='Directory containing data')
parser.add_argument('--restore_dir', default='experiment', help='Directory to restore the expriment results')
parser.add_argument('--data', default='wikidatasets')
parser.add_argument('--model', default='TransR')

parser_for_kg_wiki = parser.add_argument_group(title='wiki')
parser_for_kg_wiki.add_argument('--which', default='companies')
parser_for_kg_wiki.add_argument('--limit', default=0)

parser_for_evaluating = parser.add_argument_group(title='Evaluating')
parser_for_evaluating.add_argument('--batch_size', default=64, help='Batch size for evaluating')
parser_for_evaluating.add_argument('--ent_dim', default=20, type=int, help='Embedding dimension for Entity')
parser_for_evaluating.add_argument('--rel_dim', default=20, type=int, help='Embedding dimension for Relation')
parser_for_evaluating.add_argument('--margin', default=0.5, type=float, help='Margin for margin ranking loss')

if __name__ == '__main__':
    args = parser.parse_args()
    restore_dir = Path(args.restore_dir)
    restore_dir = restore_dir / args.data / args.which / args.model
    
    # load data
    assert args.data in ['wikidatasets', 'fb15k'], "Invalid knowledge graph dataset"
    if args.data == 'wikidatasets':
        _, kg_valid, kg_test = torchkge_ds.load_wikidatasets(which=args.which, 
                                                             limit_=args.limit, 
                                                             data_home=args.data_dir)
    elif args.data == 'fb15k':
        _, kg_valid, kg_test = torchkge_ds.load_fb15k(data_home=args.data_dir)
    
    # restore model
    assert args.model in ['TransE', 'TransR', 'DistMult'], "Invalid Knowledge Graph Embedding Model"
    if args.model == 'TransE':
        model = torchkge.models.TransEModel(args.ent_dim, kg_test.n_ent, kg_test.n_rel, dissimilarity_type='L2')
    elif args.model == 'DistMult':
        model = torchkge.models.DistMultModel(args.ent_dim, kg_test.n_ent, kg_test.n_rel, dissimilarity_type='L2')
    elif args.model == 'TransR':
        model = torchkge.models.TransRModel(args.ent_dim, args.rel_dim, kg_test.n_ent, kg_test.n_rel, dissimilarity_type='L2')
        
    checkpoint_manager = CheckpointManager(restore_dir)
    ckpt = checkpoint_manager.load_checkpoint(f'best_{args.model}.tar')
    model.load_state_dict(ckpt['model_state_dict'])
    criterion = MarginLoss(args.margin)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()
        criterion.cuda()

    sampler = BernoulliNegativeSampler(kg_test)
    test_dl = DataLoader(kg_test, batch_size=args.batch_size, use_cuda='all')

    model.eval()
    test_loss = 0
    for step, batch in enumerate(test_dl):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)
        with torch.no_grad():
            pos, neg = model(h, t, n_h, n_t, r)
            loss = criterion(pos, neg)
            test_loss += loss.item()
    test_loss /= (step+1)
    summary_manager = SummaryManager(restore_dir)
    summary_manager.load(f'summary_{args.model}.json')
    
    summary = {'test loss': round(test_loss, 4)}
    summary = dict(**summary)
    previous_summary = summary_manager.summary['Training Summary']
    previous_summary['test loss'] = round(test_loss, 4)
    summary = {'Training Summary': previous_summary}
    summary_manager.update(summary)
    
    # Link Prediction 
    lp_evaluator = LinkPredictionEvaluator(model, kg_test)
    lp_summary = lp_evaluator.evaluate(verbose=False, b_size=args.batch_size, k=10)
    lp_summary = dict(**lp_summary)
    lp_summary = {'Link Prediction': lp_summary}
    summary_manager.update(lp_summary)
    
    # Triplet Classification
    tc_evaluator = TripletClassificationEvaluator(model, kg_valid, kg_test)
    tc_evaluator.evaluate(b_size=args.batch_size)
    tc_summary = {'Accuracy': round(tc_evaluator.accuracy(b_size=args.batch_size), 4)}
    tc_summary = dict(**tc_summary)
    tc_summary = {'Triplet Classification': tc_summary}
    summary_manager.update(tc_summary)

    summary_manager.save(f'summary_{args.model}.json')

    
    
    
    
    
    
    
    