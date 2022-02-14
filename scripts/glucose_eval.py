import numpy as np
import pandas as pd
import sys 
import os
import json

def main():
	path = sys.argv[1] # result path
	k = int(sys.argv[2]) # top-k results
	for dirpath, dnames, fnames in os.walk(path):
		for f in fnames:
			if f.endswith('csv'):
				model_name = dirpath.split(";")[0].split("_")
				if len(model_name) < 3:
					continue
				xpid = model_name[0]#.split("/")[1]
				width = int(model_name[-2])
				depth = int(model_name[-1])
				df_og = pd.read_csv(os.path.join(dirpath, f))
				epoch = [i * 10 for i in range(1, 31)]
				for e in epoch:
					for proxy in ['AverageReturn', 'Test Rewards Mean']:
						df = df_og.head(e)
						df.sort_values(by=[proxy], ascending=False, inplace=True)
						s = 1 if type(df[proxy][0]) is str else 0
						rew = df[proxy][s:s+k].astype(float).tolist()
						std = df['Test Rewards Std'][s:s+k].astype(float).tolist()
						risk = df['Risk'][s:s+k].astype(float).tolist()
						magni_risk = df['MagniRisk'][s:s+k].astype(float).tolist()
						glucose = df['Glucose'][s:s+k].astype(float).tolist()
						min_bg = df['MinBG'][s:s+k].astype(float).tolist()
						max_bg = df['MaxBG'][s:s+k].astype(float).tolist()
						ins = df['Insulin'][s:s+k].astype(float).tolist()
						glen = df['GLen'][s:s+k].astype(float).tolist()
						eu = df['Euglycemic'][s:s+k].astype(float).tolist()
						hypo = df['Hypoglycemic'][s:s+k].astype(float).tolist()
						hyper = df['Hyperglycemic'][s:s+k].astype(float).tolist()
						name = f'{xpid}_episode' if proxy == 'AverageReturn' else f'{xpid}_eval'
						name += f'_{width}_{depth}_{e}.json'
						path = os.path.join(dirpath, name)
						if os.path.exists(path):
							os.remove(path)

						# Expected cost of ER treatment for hypoglycemic episode = COST_PER_HOSPITAL_VISIT * HYPOGLYCEMIC_RISK
						HOSPITAL = -1350
						# Cost of insulin = COST_PER_INSULIN_UNIT * INTERVALS_PER_YEAR * INSULIN_UNITS_PER_INTERVAL
						INSULIN = -0.32 
						INTERVALS = 12 * 24 * 365 # Five-minute intervals
						with open(path, 'w') as dump:
							json.dump({'params': [width ** 2 * depth + 48 * width], 'rew': rew, 'std': std,
										'risk': risk, 'magni': magni_risk, 'glucose': glucose,
									   'min_bg': min_bg, 'max_bg': max_bg, 'ins': ins, 'glen': glen,
									   'eu': eu, 'hypo': hypo, 'hyper': hyper, 
									   'proxy_reward': np.mean(rew), 
									   'true_reward': HOSPITAL * np.mean(hypo) + INSULIN * INTERVALS * np.mean(ins)}, dump)
						dump.close()


if __name__ == '__main__':
	main()
