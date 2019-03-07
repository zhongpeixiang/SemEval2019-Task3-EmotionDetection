import logging
import numpy as np


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}
# label_weight = [0.68, 0.13, 0.09, 0.1] # in the order: others, happy, sad and angry
# val_label_distributions = [0.85, 0.05, 0.045, 0.055] # in the order: others, happy, sad and angry
# train_whole_label_weight = [0.65, 0.14, 0.1, 0.11]
# train_whole_label_large_weight = [1.10040621, 0.38836353, 0.90516271, 1.12021303]
label_weight = {
  "train.csv": [0.65174688, 0.13944082, 0.09533021, 0.11348208],
  "train_val.csv": [0.62468681, 0.14955623, 0.10330421, 0.12245275]
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def getMetrics(predictions, ground, output_size):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. 
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    preds = predictions.argmax(axis=1)
    ground = ground.astype(int)
    labels = ground.copy()
    
    # accuracy
    accuracy = np.mean(preds==ground)
    
    # convert to one-hot
    discretePredictions = np.zeros((preds.shape[0], output_size))
    discretePredictions[np.arange(preds.shape[0]), preds] = 1
    ground = np.zeros((labels.shape[0], output_size))
    ground[np.arange(labels.shape[0]), labels] = 1
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    logging.info("True Positives per class : {0}".format(truePositives))
    logging.info("False Positives per class : {0}".format(falsePositives))
    logging.info("False Negatives per class : {0}".format(falseNegatives))
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, output_size):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        logging.info("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    logging.info("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    logging.info("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    logging.info("-> Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1