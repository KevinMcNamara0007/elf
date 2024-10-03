import os
import pickle
import onnxruntime as ort


class ClassifierManager:
    def __init__(self):
        self.classifier = None
        self.tokenizer = None

    def start_classifier(self):
        """
        Starts the ONNX classifier model with the loaded tokenizer
        :return:
        """
        try:
            with open(os.getenv("classifier_tokenizer"), 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.classifier = ort.InferenceSession(os.getenv("classifier_model"),
                                                   providers=['CUDAExecutionProvider', 'CoreMLExecutionProvider',
                                                              'CPUExecutionProvider'])
            print("Classifier model loaded.")
        except Exception as e:
            print(f'Failed to load classifier model {e}')
            exit(1)

    def shutdown_classifier(self):
        """
        Cleans up the classifier and tokenizer resources
        :return:
        """
        self.classifier = None
        self.tokenizer = None
        print("Classifier model shutting down.")