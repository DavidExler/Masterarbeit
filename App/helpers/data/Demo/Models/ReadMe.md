# Models Folder

### Expected outputs:
| Filename | Generating Tab | Explanation |
|----------|----------|----------|
| model_ABCD.pt* | Optimize | Model checkpoint of a specific classifikator at the last epoch, used to continue training after breaks |
| best_model_ABCD.pt* | Optimize | Model checkpoint of a specific classifikator  at the epoch with the highest validation accuray |
| best_pseudo_model_ABCD.pt* | Optimize | Model checkpoint of a specific classifikator  at the epoch of the pretraining with the highest validation accuray |
| pseudo_model_ABCD.pt* | Optimize | Model checkpoint of a specific classifikator  after semi supervised pretraining |
| predictions.json | Optimize | Readable predictions and labels of the annotated training and validation set of the model with the highest validation accuracy |
| predictions_ABCD.json* | Optimize | Readable predictions and labels of the annotated training and validation set of a specific model |
| pseudo_predictions_ABCD.json* | Optimize | Readable predictions and labels of the annotated training and validation set of a specific model during semi supervised pretraining |
| history_ABCD.json* | Optimiize | Radable list of the training loss, validation loss and validation accuracy at every epoch |
| pseudo_history_ABCD.json* | Optimiize | Radable list of the training loss, validation loss and validation accuracy at every epoch of the semi supervised pretraining |
| best_combo.pkl | Optimize | Python list of classification methods that resulted in the highest validation accuracy |
| val_acc_ABCD.json* | Optimize | Readable list of a validation loss and accuracy of the epoch with the highest validation accuracy |
| pseudo_val_acc_ABCD.json* | Optimize | Python list of a validation loss and accuracy of the epoch of the semi supervised pretraining with the highest validation accuracy |

[\*]: A = Encoder, B = Decoder, C = Preprocessing, D = Pretraining