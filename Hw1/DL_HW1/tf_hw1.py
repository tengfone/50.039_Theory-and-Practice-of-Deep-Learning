import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def vanillaAccuracy(y_pred, y_actual):
    total = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_actual[i]:
            total += 1
    return total / len(y_pred)


def classwiseAccuracy(y_pred, y_actual, seasons):
    totalSum = 0
    for i, eachSeason in enumerate(seasons):
        eachSum = 0
        eachClass = 0
        for j in range(len(y_actual)):
            if (y_actual[j] == i):
                if (y_actual[j] == y_pred[j]):
                    eachSum += 1
                eachClass += 1
        totalSum += eachSum / eachClass
    return totalSum / len(seasons)


def main(start_mode):
    if (start_mode == "save"):
        # Mount Files
        # label_info
        label_info_path = "C:/Users/phang/Desktop/DL_HW1/label_info.txt"

        # labels
        getLabels_path = "C:/Users/phang/Desktop/DL_HW1/gtlabels.txt"

        # Image feature folder
        img_features_path = "C:/Users/phang/Desktop/DL_HW1/imagefeatures/imagefeatures/"

        # Create a dictionary with Label:Index
        contentPage = {}

        with open(label_info_path) as f:
            for l in f:
                eachLine = l.strip().split("\t")
                contentPage[eachLine[1]] = eachLine[0]

        spring_index = int(contentPage['Spring'])
        summer_index = int(contentPage['Summer'])
        autumn_index = int(contentPage['Autumn'])

        # Create a dictionary with Label:Index
        contentPage = {}

        with open(label_info_path) as f:
            for l in f:
                eachLine = l.strip().split("\t")
                contentPage[eachLine[1]] = eachLine[0]

        spring_index = int(contentPage['Spring'])
        summer_index = int(contentPage['Summer'])
        autumn_index = int(contentPage['Autumn'])
        # Finding the correct files for Spring/Summer/Autumn
        springFiles = []
        summerFiles = []
        autumnFiles = []

        with open(getLabels_path) as f:
            for l in f:
                eachLine = l.strip().split("\t")
                eachLine = eachLine[0]
                # {FXXX-XXX-XXX-XXXXX}.jpg 0 0 0 0 1 0 0 0 0
                eachLine = eachLine.split()
                if (eachLine[spring_index + 1] == "1"):
                    springFiles.append(eachLine[0])
                if (eachLine[summer_index + 1] == "1"):
                    summerFiles.append(eachLine[0])
                if (eachLine[autumn_index + 1] == "1"):
                    autumnFiles.append(eachLine[0])

        totalFiles = springFiles + summerFiles + autumnFiles

        multi_labels = []

        # [spring,summer,autumn]

        for i in totalFiles:
            if (i in springFiles):
                multi_labels.append([1, 0, 0])
            if (i in summerFiles):
                multi_labels.append([0, 1, 0])
            if (i in autumnFiles):
                multi_labels.append([0, 0, 1])

        all_files = []

        # All files
        for i in totalFiles:
            eachSpringFile = np.load(img_features_path + i + '_ft.npy')
            all_files.append(eachSpringFile)

        all_files = np.asarray(all_files)

        # all_files will be in numpy form so like [[0.1,0.2.....],[0.5,0.1,0.3...],.....]
        # multi_labels is array [[0,0,1],[1,0,0],[0,1,0],.....]
        X_trainVal, X_test, y_trainVal, y_test = train_test_split(all_files, multi_labels, test_size=0.2,
                                                                  random_state=1, stratify=multi_labels)
        X_train, X_val, y_train, y_val = train_test_split(X_trainVal, y_trainVal, test_size=0.1875)

        # Initial Save
        np.save('train.npy', X_train)
        np.save('test.npy', X_test)
        np.save('val.npy', X_val)
        np.save('label_train.npy', y_train)
        np.save('label_val.npy', y_val)
        np.save('label_test.npy', y_test)
        np.save('Xtrain_val.npy', X_trainVal)
        np.save('ytrain_val.npy', y_trainVal)
    else:
        seasons = ['Spring', 'Summer', 'Autumn']
        kernels = ['linear', 'poly', 'rbf']
        regs = [0.01, 0.1, 0.1 ** 0.5, 1, 10 ** 0.5, 10, 100]
        recordsOfC_test = {}
        recordsOfC_val = {}
        vanStandC_test = {}
        vanStandC_val = {}
        X_train = np.load('C:/Users/phang/Desktop/DL_HW1/train.npy')
        X_test = np.load('C:/Users/phang/Desktop/DL_HW1/test.npy')
        X_val = np.load('C:/Users/phang/Desktop/DL_HW1/val.npy')
        y_train = np.load('C:/Users/phang/Desktop/DL_HW1/label_train.npy')
        y_val = np.load('C:/Users/phang/Desktop/DL_HW1/label_val.npy')
        y_test = np.load('C:/Users/phang/Desktop/DL_HW1/label_test.npy')
        X_trainVal = np.load('C:/Users/phang/Desktop/DL_HW1/Xtrain_val.npy')
        y_trainVal = np.load('C:/Users/phang/Desktop/DL_HW1/ytrain_val.npy')

        # Convert list back to Numpy Array
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        X_val = np.asarray(X_val)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        y_test = np.asarray(y_test)
        X_trainVal = np.asarray(X_trainVal)
        y_trainVal = np.asarray(y_trainVal)

        for eachKernel in kernels:
            print(f"------------------------------Running {eachKernel} kernel------------------------------")

            # Training for 3 binary SVM
            svmOutput = {"Spring": [],
                         "Summer": [],
                         "Autumn": []}

            for i, eachSeason in enumerate(seasons):
                for j in regs:
                    SVM = SVC(C=j, kernel=eachKernel, class_weight='balanced', probability=True)
                    # Splits the multi label into a binary label
                    SVM.fit(X_train, y_train[:, i].ravel())
                    svmOutput[eachSeason].append(SVM)

            for eachC, value in enumerate(regs):
                perPred_test = list()
                perPred_val = list()
                for eachSeason in seasons:
                    eachSVM = svmOutput[eachSeason][eachC]
                    SVM_pred_test = eachSVM.predict_proba(X_test)[:, 1]
                    SVM_pred_val = eachSVM.predict_proba(X_val)[:, 1]
                    perPred_test.append(SVM_pred_test)
                    perPred_val.append(SVM_pred_val)

                predLabel_test = np.argmax(perPred_test, axis=0)
                predLabel_val = np.argmax(perPred_val, axis=0)
                actualLabel_test = np.argmax(y_test, axis=1)
                actualLabel_val = np.argmax(y_val, axis=1)
                findingC = classwiseAccuracy(predLabel_test, actualLabel_test, seasons)
                eachVan = vanillaAccuracy(predLabel_test, actualLabel_test)
                eachVanVal = vanillaAccuracy(predLabel_val, actualLabel_val)
                findingCVal = classwiseAccuracy(predLabel_val, actualLabel_val, seasons)
                recordsOfC_val[value] = findingCVal
                recordsOfC_test[value] = findingC
                vanStandC_test[value] = eachVan
                vanStandC_val[value] = eachVanVal
                print(f"C value: {value} , using classwise Accuracy: {findingCVal:.4f}")

            bestC = max(recordsOfC_val, key=recordsOfC_val.get)
            bestCWAcc_test = recordsOfC_test[bestC]
            bestVanAcc_test = vanStandC_test[bestC]
            bestVanAcc_val = vanStandC_val[bestC]
            bestCWAcc_val = recordsOfC_val[bestC]
            print(f"VALIDATION: {recordsOfC_val}")
            print(f"TEST: {recordsOfC_test}")
            print("\n----Report-----")
            print(
                f"Best performance of C in class wise accuracy(Test score) is when c = {bestC} of value {bestCWAcc_test:.4f} for train set only")
            print(
                f"Best performance of C in class wise accuracy(Val score) is when c = {bestC} of value {bestCWAcc_val:.4f} for train set only")
            print(f"Using C = {bestC}, applying the Vanilla Accuracy(Test score) will get you: {bestVanAcc_test:.4f}")
            print(f"Using C = {bestC}, applying the Vanilla Accuracy(Val score) will get you: {bestVanAcc_val:.4f}")

            print(f"\n---Retraining SVM Model with best C value of {bestC} and train + validation set---")
            for i, eachSeason in enumerate(seasons):
                for j in regs:
                    SVM = SVC(C=bestC, kernel=eachKernel, class_weight='balanced', probability=True)
                    SVM.fit(X_trainVal, y_trainVal[:, i].ravel())
                    svmOutput[eachSeason].append(SVM)

            latestSVM = len(svmOutput["Spring"]) - 1

            finalPred_test = list()
            finalPred_val = list()
            for eachSeason in seasons:
                eachSVM = svmOutput[eachSeason][latestSVM]
                SVM_pred_test = eachSVM.predict_proba(X_test)[:, 1]
                SVM_pred_val = eachSVM.predict_proba(X_val)[:, 1]
                finalPred_test.append(SVM_pred_test)
                finalPred_val.append(SVM_pred_val)
            predLabel2 = np.argmax(perPred_test, axis=0)
            predLabel2_val = np.argmax(perPred_val, axis=0)
            newCWAccuracy = classwiseAccuracy(predLabel2, actualLabel_test, seasons)
            newVanAccuracy = vanillaAccuracy(predLabel2, actualLabel_test)
            newCWAccuracyVAL = classwiseAccuracy(predLabel2_val, actualLabel_val, seasons)
            newVanAccuracyVAL = vanillaAccuracy(predLabel2_val, actualLabel_test)
            print(f"New Class Wise Accuracy(TEST score): {newCWAccuracy:.4f}")
            print(f"New Vanilla Accuracy(TEST score): {newVanAccuracy:.4f}\n")
            print(f"New Class Wise Accuracy(VAL score): {newCWAccuracyVAL:.4f}")
            print(f"New Vanilla Accuracy(VAL score): {newVanAccuracyVAL:.4f}\n")


if __name__ == '__main__':
    # "save" for saving numpy, "load" is to load from numpy
    start_mode = "load"
    main(start_mode)
