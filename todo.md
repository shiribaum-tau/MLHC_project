Todo:

* More models:
    * Add blood tests

* Data:
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


Try with best MLP/TF arch (retrain w/ 100 epochs):
* all individuals -> skin cancer (TF->V)
* all individuals -> breast cancer (TF->V)
* all individuals -> diabetes (TF->V)
* all individuals -> pancreatic (TF->V)


