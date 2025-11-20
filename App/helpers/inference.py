import numpy as np
import os
import pickle
import torch
import json
#
# takes masks, images (Z,X,Y,C)?, label_store.json
# returns best_combo, predicted, checkpoints
#

from helpers.classifier_helper import ClassificatorBlobHelper, get_model, bicubic_upsample_3d, preprocess_blob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def inference_combo(in_folder, in_folder_images, out_folder):
    path = os.path.join(BASE_DIR, 'data', in_folder, 'best_combo.pkl')
    with open(path, 'rb') as f:
        (enc, dec, pre, pretrain) = pickle.load(f) 
    

    infer(enc, dec, pre, pretrain, in_folder, in_folder_images, out_folder)
    return 1
    
def infer(encoder, decoder, preprocessor, pretraining, in_folder, in_folder_images, out_folder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classificator_helper = ClassificatorBlobHelper()
    classificator_helper.reload_images(in_folder_images)
    classificator_helper.lum = len(np.unique(classificator_helper.mask))
    model, _, _ = get_model(encoder, decoder, preprocessor, pretraining, None, None, inference=True)
    if pretraining == "Kein Vortraining":
            print("removing space")
            pretraining = "KeinVortraining"
    if os.path.exists(os.path.join(BASE_DIR, 'data', in_folder, f"best_model_{encoder}{decoder}{pretraining}{preprocessor}.pt")):
        checkpoint = torch.load(os.path.join(BASE_DIR, 'data', in_folder, f"best_model_{encoder}{decoder}{pretraining}{preprocessor}.pt"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights from checkpoint.")
    else:
        raise FileNotFoundError(f"No checkpoint found for model best_model_{encoder}{decoder}{pretraining}{preprocessor}.pt in {in_folder}")
    print(preprocessor)
    if preprocessor == 'Segmentierungsmaske':
        in_channel_dist = False
    else:
        in_channel_dist = True

    model = model.to(device)
    model.eval()
    preds = {}
    overflow = False
    img_idx = 0
    blob_idx = 0
    while overflow == False:
        if blob_idx >= classificator_helper.lum:
            img_idx += 1
            blob_idx = 0
            classificator_helper.lum = len(np.unique(classificator_helper.mask))
        blob, overflow = classificator_helper.get_blob(image_index=img_idx, blob_index=blob_idx, in_channel_dist=in_channel_dist)
        if blob is None:
            break
        blob = bicubic_upsample_3d(blob, (256, 256))
        blob = blob.float()
        blob = preprocess_blob(blob)
        blob_batched = blob.unsqueeze(0).to(device)
        pred = model(blob_batched).detach().cpu().numpy()
        print(pred)
        pred_label = np.argmax(pred, axis=1)
        print(pred_label)
        if img_idx not in preds:
            preds[img_idx] = {}
        preds[img_idx][blob_idx] = int(pred_label[0])
        blob_idx += 1
        i += 1
    with open(os.path.join(BASE_DIR, 'data', out_folder, 'predicted.json'), 'w') as f:
        json.dump(preds, f)
    return preds