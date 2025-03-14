from py_experimenter.experimenter import PyExperimenter

experimenter = PyExperimenter(
    experiment_configuration_file_path="config-classification.yaml",
    use_codecarbon=False
)
experimenter.fill_table_from_config()

#experimenter = PyExperimenter(
#    experiment_configuration_file_path="config/experiments-fullforests-regression.cfg",
#    use_codecarbon=False
#)
#experimenter.fill_table_from_config()