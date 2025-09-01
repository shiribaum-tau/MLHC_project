Todo:
* Metrics:
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

* Write:
    * Intro
    * About original model
    * data (about UKB, and about our subsample of it)
    * our changes
    * results
    * discussion/limitations

    * the loss is masked such that only the first instance of cancer affects the loss.
        It *is* calculated as part of the AUC, etc.



Things to try:
* tuning metric = val loss
* more epochs + early stop
    * early stop on learning rate and tuning metric
* add men to the data (halleluja)
* Pancreatic cancer
    * Skin cancer?
* pick threshold (maximize f1?)

