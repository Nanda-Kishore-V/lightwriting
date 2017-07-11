import yaml

from constants_env import (
	HOME,
	ROS_WS,
	)

def main():
	
	with open(HOME + 'data/all49.yaml') as infile:
		crazyflies = yaml.load(infile)['crazyflies']

	ids = map(lambda x: crazyflies[int(x) - 1], raw_input().split())
	with open(HOME + 'data/ground_positions.yaml', 'w') as outfile:
		yaml.dump({'crazyflies': ids}, outfile)
	
	with open(ROS_WS + 'launch/crazyflies.yaml', 'w') as outfile:
		yaml.dump({'crazyflies': ids}, outfile)


if __name__=='__main__':
	main()