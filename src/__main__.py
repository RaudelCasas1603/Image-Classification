import argparse
import json
import os

# Import the modules
from .dataset import Collector, DataSetCreate
from .model import Model, LoadModel, DEFAULT_MODEL_FILE
from .features import Transcripter

def main():
    parser = argparse.ArgumentParser()

    # Add the command argument
    parser.add_argument(
        "command",
        choices=["collect-data", "build-dataset", "train", "transcript", "tune"],
        help="The command to execute. Choose from 'collect-data', 'build-dataset', 'train', or 'transcript'."
    )
    
    # For the collector of the data
    parser.add_argument(
        "-n", "--amount-pictures",
        nargs = '?',
        type = int,
        metavar = "N",
        help = "The number of pictures to generate per class."
    )
    
    parser.add_argument(
        "-d", "--directory",
        nargs = '?',
        metavar="PATH",
        help="The directory to store the collected data."
    )

    parser.add_argument(
        "-c", "--classes",
        nargs = '?',
        metavar="PATH",
        help = "The json file with the labels to classify."
    )

    parser.add_argument(
        "-w", "--webcam",
        nargs = '?',
        type = int,
        metavar = "N",
        help = "The webcam device number."
    )

    # Flags for build-dataset
    parser.add_argument(
        "-f", "--file-dataset",
        nargs = '?',
        metavar="PATH",
        help = "The output file where the dataset will be sotred."
    )

    # Flags for the train 
    parser.add_argument(
        "-m", "--file-model",
        nargs = '?',
        metavar="PATH",
        help = "The output file where the model brain will be sotred."
    )

    # Flags to run the demo transcript
    parser.add_argument(
        "-o", "--transcript-out",
        nargs = "?",
        metavar = "PATH",
        help = "The output from the transcription"
    )
    
    args = parser.parse_args()
    contex = {}
    # Parse the arguments
    if args.command == "collect-data":
        # Load the new classes
        class_map_dict = json.load(open(args.classes))
        contex["classes"] = class_map_dict
        contex["amount_classes"] = len(class_map_dict)
        
        if args.amount_pictures:
            contex["amount_pics"] = args.amount_pictures
        if args.directory:
            contex["directory"] = args.directory
        if args.webcam:
            contex["device"] = args.webcam

        # Unpack the configuration 
        collector = Collector(**contex)
        collector.start()       # Start collecting the data
        print("Data collection completed")
        
    elif args.command == "build-dataset":
        if args.file_dataset:
            contex["filename"] = args.file_dataset
            
        if args.directory:
            contex["directory"] = args.directory

        # Unpack the configuration
        dataset = DataSetCreate(**contex)
        dataset.build()
        dataset.save()
        print("Dataset builded")
        
    elif args.command == "train":
        if args.file_model:
            contex["modelfile"] = args.file_model
            
        if args.file_dataset:
            contex["dataset"] = args.file_dataset

        if os.path.exist(contex["modelfile"]):
            model = LoadModel(contex["modelfile"])  #  To retrain the model
        else:
            model = Model(**contex)
            
        model.train()
        model.test()
        model.save()
        print("Model trained")

    elif args.command == "tune":  # To tune the model with better performance

        if args.file_model:
            contex["modelfile"] = args.file_model
            
        if args.file_dataset:
            contex["dataset"] = args.file_dataset


        if not os.path.exists(DEFAULT_MODEL_FILE \
                              if not "modelfile" in contex \
                              else contex["modelfile"]):
            raise Exception("To tune a model you must to have a model.pickle file or something")
        
        model = LoadModel(**contex)
        model.fine_tuning()     # Do the tuning stuff
        model.test()
        model.save()
        print("Model tuned")
    elif args.command == "transcript":
        if args.file_model:
            contex["modelfile"] = args.file_model

        if args.classes:          # If you have custom classes
            # Load the new classes
            class_map_dict = json.load(open(args.classes))
            contex["classes"] = class_map_dict

        if args.webcam:
            contex["device"] = args.webcam
            
        transcripter = Transcripter(**contex)
        transcripter.transcript()  # Run the transcript
    else:
        return -1
    
    return 0                    # As success

if __name__ == "__main__":
    exit(main())
