# Transfer Learning

Documenting transfer learning options

## Using weights from a previous model (feature-extraction)

Take a model from CAS, and use the weights to train in ETZ.

- preprocess with CAS
- train with CAS
- get model filename from CAS
- set transfer learning variables in new config for ETZ
- preprocess with ETZ
- train with ETZ

## Using weights from a previous model, train some layers (fine-tuning)

- preprocess with CAS
- train with CAS
- get model filename from CAS
- set transfer learning variables in new config for ETZ
- lower learning rate


## Use autoencoder as weights, train decoder

TBD