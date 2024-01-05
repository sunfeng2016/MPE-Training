map_param_registry = {
	"10vs10": {
		"n_reds": 10,
		"n_blues": 10,
		"limit": 200
	},
	"50vs50": {
		"n_reds": 50,
		"n_blues": 50,
		"limit": 500
	}
}


def get_map_params(map_name):
	return map_param_registry[map_name]
