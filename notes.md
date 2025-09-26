# Issues
## 1. intermediate NaN in model
### Description
When using a dataset that was low pass filtered at 100Hz, multiple components of the RNN would produce outputs of all NaN values. This did not transfer over to my local machine and the baseline dataset (resampled to 128Hz, high-pass at 0.5Hz) did not have this issue on OSC. Clamping values only resulted in many outputs being 0.
### Fixes
Used norm first in the transformer layers to normalize values before passed to ops that could exceed precision

## 2. OOM errors for high SR EEG data
### Description
When using EEG datasets with a high sampling rate (512Hz specifically) system memory would be exceeded due to each dataloader subprocess copying the large dataset and cause OOM_KILL errors.
### Fixes
Implemented a multi-threading specific dataloader that utilizes a shared instance of the data for multiple loading processes