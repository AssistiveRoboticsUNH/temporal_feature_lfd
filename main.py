


if __name__ == '__main__':

	from parameter_parser import parse_model_args
	lfd_params = parse_model_args()

	from model.model import LfDNetwork
	net = LfDNetwork(lfd_params, is_training = True)

	filename = train(lfd_params, net)

	lfd_params.args.base_modelname = filename
	net = LfDNetwork(lfd_params, is_training = False)

	eval(lfd_params, net)