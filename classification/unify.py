import os
import pandas as pd

class Unify:
    '''
    Unifies data from same participants across multiple files.
    '''

    def __init__(self, root_dir):        
        doubles = {
            'PS01': 'PS02',
            'PS06': 'PS07',
            'PS11': 'PS12',
            'PS13': 'PS14',
            'PS15': 'PS16',
            'PS22': 'PS23',
            'PS24': 'PS25',
        }

        # make dir root/unified if it doesn't exist
        unified_dir = os.path.join(root_dir, 'unified')
        os.makedirs(unified_dir, exist_ok=True)

        # list of existing files
        existing_files = os.listdir(root_dir)
        existing_files = sorted([f for f in existing_files if f.endswith('.csv')])
        unprocessed = existing_files.copy()

        # participant index
        i = 1

        # go through the files and concatenate dataframes
        while len(unprocessed) > 0:
            file = unprocessed[0]
            part_name = file.split('.')[0]
            print(f'Processing participant: {part_name}')
            if part_name in doubles.keys():
                unprocessed.remove(file)
                unprocessed.remove(f'{doubles[part_name]}.csv')
                
                # concatenate dataframes
                file1 = os.path.join(root_dir, f'{part_name}.csv')
                file2 = os.path.join(root_dir, f'{doubles[part_name]}.csv')
                
                # read dataframes
                df1 = pd.read_csv(file1)
                df2 = pd.read_csv(file2)
                unified_df = pd.concat([df1, df2], ignore_index=True)
                
                # save unified dataframe
                unified_file = os.path.join(unified_dir, f'PS{i:02d}.csv')
                unified_df.to_csv(unified_file, index=False)
                print(f'Unified file created: {unified_file}')
                
            else:
                unprocessed.remove(file)
                
                # read dataframe
                file_path = os.path.join(root_dir, file)
                df = pd.read_csv(file_path)
                
                # save unified dataframe
                unified_file = os.path.join(unified_dir, f'PS{i:02d}.csv')
                df.to_csv(unified_file, index=False)
                print(f'Single file created: {unified_file}')
            
            i += 1