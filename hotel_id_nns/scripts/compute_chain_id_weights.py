import pandas as pd
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent

def main():

    pd.set_option('display.max_rows', None)

    train_ds = pd.read_csv(root_dir / "data/dataset/hotel_train_chain.csv", names=['path', 'chain_id'], sep=' ')
    test_ds = pd.read_csv(root_dir / "data/dataset/hotel_test_chain.csv", names=['path', 'chain_id'], sep=' ')

    ds = pd.concat((train_ds, test_ds))


    num_chain_id = ds.value_counts('chain_id')
    weights = (1 / num_chain_id) 
    weights = weights / weights.sum()
    weights.name = "weights"
    print(weights)

    weights.to_csv("data/dataset/chain_id_weights.csv")


if __name__ == "__main__":
    main()