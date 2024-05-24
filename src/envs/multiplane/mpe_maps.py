map_param_registry = {
	"5vs5": {
		"n_reds": 5,
		"n_blues": 5,
		"limit": 100,
		"size_x": 800,
		"size_y": 800,
		"height": 4,
		"init_dis": 500,
		"red_y_range": 200,
		"blue_y_range": 200
	},
	"10vs10": {
		"n_reds": 10,
		"n_blues": 10,
		"limit": 200,
		"size_x": 800,
		"size_y": 800,
		"height": 4,
		"init_dis": 500,
		"red_y_range": 200,
		"blue_y_range": 200
	},
	"50vs50": {
		"n_reds": 50,
		"n_blues": 50,
		"limit": 400,
		"size_x": 1200,
		"size_y": 1200,
		"height": 4,
		"init_dis": 1000,
		"red_y_range": 600,
		"blue_y_range": 600
	},
	"100vs100": {
		"n_reds": 100,
		"n_blues": 100,
		"limit": 600,
		"size_x": 1500,
		"size_y": 1500,
		"height": 4,
		"init_dis": 1000,
		"red_y_range": 1000,
		"blue_y_range": 1000
	}
}


def get_map_params(map_name):
	return map_param_registry[map_name]
