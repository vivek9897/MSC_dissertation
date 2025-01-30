import os
import pandas as pd

pairs = ['AUDUSD', 'EURGBP', 'EURUSD', 'USDCAD']
for pair in pairs:
	print(pair)

	dir_path = f'./Monthly/{pair}'

	# get data files
	data_files = os.listdir(dir_path)
	files = [f for f in data_files if os.path.isfile(os.path.join(dir_path, f))]

	df = pd.DataFrame()
	for file in files:
		file_df = pd.read_csv(os.path.join(dir_path, file), names=['Pair', 'Timestamp', 'Bid', 'Ask'])
		df = pd.concat([df, file_df])

	df.to_csv(f'./Concatenated/{pair}_concatenated.csv', index=False)
