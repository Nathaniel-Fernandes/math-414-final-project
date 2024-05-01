import pickle
from thirdparty.malwaredetection.Extract.PE_main import extract_infos
from classifier import get_benign_data, get_malware_data, create_custom_dataset
import os
import datetime
import torch
import pdb


# Run "malware-detection"
def malware_detection(filepaths: list):
    filepaths = [f"data/DikeDataset/files/{path[12:-4]}.exe" for path in filepaths]

    os.makedirs("runs", exist_ok=True)
    run_name = f"runs/malwaredetection_{str(datetime.datetime.now())}.csv"

    # Loading the classifier.pkl and features.pkl
    with open(
        "thirdparty/malwaredetection/Classifier/PE/pickel_malware_detector.pkl", "rb"
    ) as file:
        clf = pickle.load(file)
    file.close()
    with open("thirdparty/malwaredetection/Classifier/PE/features.pkl", "rb") as file2:
        features = pickle.load(file2)
    file2.close()

    with open(run_name, "w") as file:
        file.write(",".join(["filename", "pred", "true", "raw_pred"]))
        file.write("\n")

        for filepath in filepaths:
            try:
                executable_path = (
                    f"data/DikeDataset/files/{filepath[12:-4]}.exe"
                    if filepath[-3:] == "txt"
                    else filepath
                )

                # extracting features from the PE file mentioned in the argument
                data = extract_infos(executable_path)

                # matching it with the features saved in features.pkl
                pe_features = list(map(lambda x: data[x], features))
                # print("Features used for classification: ", pe_features)

                # prediciting if the PE is malicious or not based on the extracted features
                res = clf.predict([pe_features])[0]
                output = ["malware", "benign"][res]

                file.write(
                    ",".join(
                        [
                            filepath,
                            output,
                            "benign" if "benign" in filepath else "malware",
                            str(clf.predict_proba([pe_features])[0][0]),
                        ]
                    )
                )
                file.write("\n")
            except Exception as e:
                print(f"There was an error in this file: {filepath}: {e}")
                # pdb.set_trace()


if __name__ == "__main__":

    # 0 = benign, 1 = malware
    load_saved = True
    if load_saved == False:
        print("loading from files")

        data, labels = get_benign_data([], [])
        data, labels = get_malware_data(data, labels)

        train_dataset, val_dataset, train_loader, val_loader, device = (
            create_custom_dataset(data, labels)
        )
        torch.save(
            [train_dataset, val_dataset, train_loader, val_loader, device],
            "saved_data.pt",
        )

    else:
        print("loading presaved")
        train_dataset, val_dataset, train_loader, val_loader, device = torch.load(
            "saved_data.pt"
        )

    validation_files = [d[1][1] for d in val_dataset]

    malware_detection(validation_files)
