from py_experimenter.experimenter import PyExperimenter


if __name__ == "__main__":
    pe = PyExperimenter(experiment_configuration_file_path=f"config/experiments.yaml")
    pe.fill_table_from_config()