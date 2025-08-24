Todo:
* Metrics:
    * AUC
    * AUPR
    * F1

* More models:
    * Transformer from paper
    * GRU from paper?
    * Gridsearch for params
    * Add blood tests
    * Add heirarchy

* Data:
    * Inclusion/Exclusion criteria (w/ men?, blood tests for pregnant people/people with cancer?)
    * Diseases to exclude?
    * 250k people

* Write:
    * Intro
    * About original model
    * data (about UKB, and about our subsample of it)
    * our changes
    * results
    * discussion/limitations

    * the loss is masked such that only the first instance of cancer affects the loss.
        It *is* calculated as part of the AUC, etc.
