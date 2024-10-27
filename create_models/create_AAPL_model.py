from src.Classes.ModelCreator import ModelCreator


creator=ModelCreator("AAPL",".h5",["1m","1h"])
creator.train_tune(plot=True)
