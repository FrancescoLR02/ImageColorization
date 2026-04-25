import pandas as pd
import numpy as np
import sys


#!Take as input the result file from the poll and then combine the human choice with the csv log file


if len(sys.argv) > 1:
   pollDataset = sys.argv[1]
else:
   sys.exit('Provide the file downloaded from the poll!')

df = pd.read_csv(pollDataset)
entireDF = pd.read_csv('DPO_dataset/CSV_log.csv')

def GetMajorityVotes(votes):
   counts = votes.value_counts()
   
   # If there is more than one option voted for AND the top two are a tie
   if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
      return None
      
   # Otherwise, return the most popular choice
   return counts.index[0]

# --- MODIFIED: Use .agg() to run the function on both columns at once ---
final_df = df.groupby('image_index').agg({
    'humanChoice': GetMajorityVotes,
    'rejectionChoice': GetMajorityVotes
}).reset_index()

# Count and remove the exact ties
# (If humanChoice is a tie, rejectionChoice is also a tie, so we only need to check one)
tied_count = final_df['humanChoice'].isna().sum()
final_df = final_df.dropna(subset=['humanChoice'])

# Sort the dataframe neatly from lowest index to highest
final_df = final_df.sort_values(by='image_index').reset_index(drop=True)

# --- MODIFIED: Prepare BOTH columns in entireDF to accept strings ---
entireDF['humanChoice'] = entireDF['humanChoice'].astype(object)

# Just in case rejectionChoice doesn't exist in entireDF yet, create it safely
if 'rejectionChoice' not in entireDF.columns:
    entireDF['rejectionChoice'] = None
entireDF['rejectionChoice'] = entireDF['rejectionChoice'].astype(object)

# Loop through and map BOTH choices to the master dataset
for i in range(len(final_df)):
   idx = final_df.loc[i, 'image_index']
   
   # Assign the winner
   entireDF.loc[idx, 'humanChoice'] = str(final_df.loc[i, 'humanChoice'])
   
   # Assign the loser
   entireDF.loc[idx, 'rejectionChoice'] = str(final_df.loc[i, 'rejectionChoice'])

# Save the clean, finalized dataset (Added index=False to keep it clean!)
entireDF.to_csv('FinalResults.csv', index=False)

print(f"Saved final preferences to FinalResults.csv!")