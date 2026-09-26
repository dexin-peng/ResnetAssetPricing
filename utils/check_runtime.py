"""Check training prerequisites and output portability without launching a job."""
from pathlib import Path
import argparse
import ast
import importlib.util

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.project_root.resolve();problems=[]
    tree=ast.parse((root/'config/base.py').read_text())
    io=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IOConfig')
    base=next(n for n in io.body if isinstance(n,ast.FunctionDef) and n.name=='base_dir')
    absolute=[n.value for n in ast.walk(base) if isinstance(n,ast.Constant) and isinstance(n.value,str) and Path(n.value).is_absolute()]
    if absolute:
        problems.append('config/base.py: IOConfig.base_dir contains an absolute machine path. Make it resolve from the current checkout before training; the current configuration can write to another project.')
    for name in ['datashare_with_return.pkl','PredictorData2024.xlsx','F-F_Research_Data_5_Factors_2x3.csv']:
        if not (root/'source_data'/name).is_file():problems.append(f'Missing licensed/training input: source_data/{name}')
    for name in ['torch','ml_collections','statsmodels','tensorboard','openpyxl']:
        if importlib.util.find_spec(name) is None:problems.append(f'Missing training dependency: {name}')
    if problems:
        print('\n'.join('FAIL '+problem for problem in problems));raise SystemExit(1)
    print('PASS: portable output configuration, training inputs and dependencies. No model was trained.')
if __name__=='__main__': main()
