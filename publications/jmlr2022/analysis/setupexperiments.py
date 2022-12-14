from py_experimenter.experimenter import PyExperimenter

experimenter = PyExperimenter(experiment_configuration_file_path="config/experiments-analysis-classification.cfg")
experimenter.fill_table_from_config()

experimenter = PyExperimenter(experiment_configuration_file_path="config/experiments-analysis-regression.cfg")
experimenter.fill_table_from_config()
