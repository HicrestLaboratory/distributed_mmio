from csv import Error
from pathlib import Path
import pprint
import sbatchman as sbm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from io import StringIO

GPUPERNODE=2

def plot_csv(df: pd.DataFrame, title: str, out_file: Path):
  # Add the new columns
  df['nodeid'] = df['rank'] // GPUPERNODE
  df['gpuid'] = df['rank'] % GPUPERNODE

  # Create the point plot
  plt.figure(figsize=(10, 8))
  sns.scatterplot(data=df, x='colid', y='rowid', hue='nodeid', style='gpuid', palette='viridis', s=100)

  # Set plot title and labels
  plt.title(title)
  plt.xlabel('Column ID')
  plt.ylabel('Row ID')
  plt.legend(title='Rank')
  plt.grid(True)

  # Save the plot to a file
  plt.savefig(out_file)
  print(f"Plot saved to {out_file}")
  
def get_csv_from_stdout(stdout: str):
  i = 0
  lines = stdout.splitlines()
  csv_lines = []
  
  while lines[i] != '[OUT CSV]':
    i += 1
  i += 1
  while lines[i] != '[END OUT CSV]':
    csv_lines.append(lines[i])
    i += 1

  csv_str = '\n'.join(csv_lines)
  df = pd.read_csv(StringIO(csv_str), skipinitialspace=True)
  return df

if __name__ == "__main__":
  jobs = sbm.jobs_list()
  jobs_failed = []
  for j in jobs:
    if j.status == sbm.Status.COMPLETED.value:
      stdout = j.get_stdout()
      if not stdout:
        raise Error(f'Job {j} has empty stdout!')
      
      df = get_csv_from_stdout(stdout)
      print(j)
      _, __, args = j.parse_command_args()
      if not args:
        raise Error(f'Job {j} wrong args!')
      
      ranks = args['np']
      file = Path(args['f'])
      grid_rows = args['r']
      grid_cols = args['c']
      partitioning_type = args['p']
      transpose = bool(args['t'])
      # The following will only work for PaRMAT names
      file_parts = file.stem.split('_')
      N = int(file_parts[1][1:])
      M = int(file_parts[2][1:])
      print(args)
      print(df)
      print('='*50)
      name = f'{N}x{N}_{M}nz_{ranks}ranks_{grid_rows}x{grid_cols}grid_{partitioning_type}part{"_transpose" if transpose else ""}'
      path = Path(f'results/{name}.png')
      path.parent.mkdir(exist_ok=True, parents=True)
      plot_csv(df, name, path)
    else:
      jobs_failed.append(j)
      
  if jobs_failed:
    print('!! FAILED jobs:')
    for j in jobs_failed:
      pprint.pprint(j)
