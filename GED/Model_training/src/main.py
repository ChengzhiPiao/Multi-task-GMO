from utils import tab_printer
from trainer_IMDB import Trainer as Trainer_IMDB
from trainer import Trainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    
    if args.dataset in ["IMDBMulti"]:
        trainer = Trainer_IMDB(args)
    else:
        trainer = Trainer(args)

    if args.model_epoch_start > 0:
        trainer.load(args.model_epoch_start)
            
    # args.learning_rate*=2
    args.learning_rate/=2
    if args.model_train == 1:
        for epoch in range(args.model_epoch_start, args.model_epoch_end):
            trainer.cur_epoch = epoch
            trainer.fit()
            trainer.save(epoch + 1)
            #trainer.score('val')
            trainer.score('test')
            # if epoch % 10 == 9:
            #     args.learning_rate*=0.8
            #if not args.demo:
             #   trainer.score('test2')
    else:
        trainer.cur_epoch = args.model_epoch_start
        # trainer.score('test', test_k=100)
        trainer.score_save('test')
        # trainer.batch_score('test', test_k=100)
        """
        test_matching = True
        trainer.cur_epoch = args.model_epoch_start
        #trainer.score('val', test_matching=test_matching)
        trainer.score('test', test_matching=test_matching)
        #if not args.demo:
         #   trainer.score('test2')
        """

if __name__ == "__main__":
    main()
