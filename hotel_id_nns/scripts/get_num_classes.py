import pandas as pd


def main():
    train_ds = pd.read_csv("data/dataset/hotel_train_chain.csv", names=['path', 'chain_id'], sep=' ')
    test_ds = pd.read_csv("data/dataset/hotel_test_chain.csv", names=['path', 'chain_id'], sep=' ')

    ds = pd.concat((train_ds, test_ds))

    print(ds.max())

    

if __name__ == "__main__":
    main()