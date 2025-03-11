from py_experimenter.experimenter import PyExperimenter


if __name__ == "__main__":
    for approach in ["bootstrapping", "parametricmodel", "databaseperparameter"]:
        pe = PyExperimenter(experiment_configuration_file_path=f"config/{approach}.yaml")
        pe.fill_table_from_config()