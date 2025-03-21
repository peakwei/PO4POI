You should run this code in a pycharm Interpreter
To run default dataset nyc, just place api and base(in main) with your owns
To run TKY dataset, you should change the amount of 'eval_data' randomly sampled by split_data function in Manager to 2000, and 'sample_data' size (which in Evaluator UCB_evaluation) to 512.
