from sklearn.metrics import f1_score 
import pandas as pd 
import argparse
from pathlib import Path

def evaluation_marco_f1(ansfile:str, predictfile:str) -> float:

    def process_submit_file_M(src: str, type_to_col_name: str):
        p = pd.read_csv(src)
        p[type_to_col_name] = p['Type'].apply(lambda x: int(x[-1]))
        return p

    # Process the files
    ans = process_submit_file_M(ansfile, type_to_col_name='gt')
    predict = process_submit_file_M(predictfile, type_to_col_name='pred')

    # Create a mapping from filename to prediction
    filename_to_pred = predict.set_index('filename')['pred'].to_dict()

    # Map the predictions to the ans DataFrame
    ans['pred'] = ans['filename'].map(filename_to_pred)

    # Calculate the F1 score
    f1 = f1_score(ans['gt'], ans['pred'], average='macro')
    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ans", type=Path, default=Path("dataset")/"origin_ans.csv")
    parser.add_argument("--predict", type=Path)
    args = parser.parse_args()
    f1 = evaluation_marco_f1(
            ansfile=args.ans,
            predictfile=args.predict
        )
    print(f"macro F1-score : {f1}")