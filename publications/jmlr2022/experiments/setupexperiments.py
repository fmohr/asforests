from py_experimenter.experimenter import PyExperimenter

experimenter = PyExperimenter(config_file="config/experiments.cfg")
#experimenter.reset_experiments("done")
experimenter.fill_table_from_config()