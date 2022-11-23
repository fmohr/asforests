from py_experimenter.experimenter import PyExperimenter

experimenter = PyExperimenter(experiment_configuration_file_path="config/experiments-analysis.cfg")
#experimenter.reset_experiments("error")
#experimenter.reset_experiments("running")
experimenter.fill_table_from_config()
