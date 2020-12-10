import numpy as np
import os
import cv2
import pandas as pd


def clahe(bgr_image, gridsize=2):
    # convert image to HSV format
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    #split to channels
    hsv_planes = cv2.split(hsv)

    #create CLAHE using cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))

    # apply clahe on V channel
    hsv_planes[2] = clahe.apply(hsv_planes[2])

    # merge channels
    hsv = cv2.merge(hsv_planes)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def hist_equ(image):
    # convert image to YUV format from RGB
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return image

def organize_data(save=False, size=256, data="all", label="diagnosis-2class", show=False, color="GRY"):
    # label = diagnosis-2class or abnormality

    X_Train_arr = []
    X_Test_arr = []

    Y_Train, Y_Test = np.empty([0]), np.empty([0])
    Y_train, Y_test = organize_labels(label=label)

    path = r'D:\Users\Burak\mammograpgy\jpg'
    for dir in os.listdir(path):
        try:
            cropdir = dir[8:]
            if (cropdir[-1].isdigit()):
                print(cropdir)
                abnormality_id = cropdir[-1]
                p_index = cropdir.find('P')
                patient_id = cropdir[p_index + 2: p_index + 7]
                breast = 'LEFT' if 'LEFT' in cropdir else 'RIGHT'
                view = 'MLO' if 'MLO' in cropdir else 'CC'
                abnormality = 'calcification' if cropdir[0:4] == 'Calc' else 'mass'
                is_train = True if 'Training' in cropdir else False
                id = 'P_' + patient_id + '_' + breast + '_' + view + '_' + abnormality_id + '_' + abnormality

                if len(os.listdir(path + '\\' + dir)) == 0:
                    continue
                dir_ = str(os.listdir(path + '\\' + dir)[0])

                if len(os.listdir(path + '\\' + dir + '\\' + dir_)) == 0:
                    continue
                subdir = str(os.listdir(path + '\\' + dir + '\\' + dir_)[0])
                print('     ' + subdir)

                if len(os.listdir(
                        path + '\\' + dir + '\\' + str(os.listdir(path + '\\' + dir)[0]) + '\\' + subdir)) == 0:
                    continue
                subsubdir = str(
                    os.listdir(path + '\\' + dir + '\\' + str(os.listdir(path + '\\' + dir)[0]) + '\\' + subdir)[0])
                print('             ' + subsubdir)

                imagew = cv2.imread(path + '\\' + dir + '\\' + dir_ + '\\' + subdir + '\\' + subsubdir)

                image = clahe(imagew)
                # image = hist_equ(imagew)


                image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

                if(color == "GRY"):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if(show):
                    cv2.imshow(id, image)
                    cv2.waitKey(500)

                if(label == 'diagnosis-2class'):
                    if is_train:
                        if (Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0] == 1 or
                                Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0] == 0):
                            X_Train_arr.append(image)
                            Y_Train = np.append(Y_Train, Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0])

                    else:
                        if (Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0] == 1 or
                                Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0] == 0):
                            X_Test_arr.append(image)
                            Y_Test = np.append(Y_Test, Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0])

                elif(label == 'abnormality'):
                    if is_train:
                        if abnormality == 'calcification':
                                X_Train_arr.append(image)
                                Y_Train = np.append(Y_Train, 0)
                        else:

                                X_Train_arr.append(image)
                                Y_Train = np.append(Y_Train, 1)
                    else:
                        if abnormality == 'calcification':
                                X_Test_arr.append(image)
                                Y_Test = np.append(Y_Test, 0)
                        else:
                                X_Test_arr.append(image)
                                Y_Test = np.append(Y_Test, 1)

                else:
                    print("*********** INVALID LABEL SET! **********")

        except Exception as err:
            print("***  Exception!  ****")
            print(err)

    X_Train = np.array(X_Train_arr)
    X_Test = np.array(X_Test_arr)

    print(X_Train.shape)
    print(X_Test.shape)

    np.save('data/' + label + '/X_train.npy', X_Train)
    np.save('data/' + label + '/X_test.npy', X_Test)
    np.save('data/' + label + '/Y_train.npy', Y_Train)
    np.save('data/' + label + '/Y_test.npy', Y_Test)

def organize_labels():
        Calc_test_labels = pd.read_csv('csv/calc_case_description_test_set.csv')
        Calc_train_labels = pd.read_csv('csv/calc_case_description_train_set.csv')
        Mass_test_labels = pd.read_csv('csv/mass_case_description_test_set.csv')
        Mass_train_labels = pd.read_csv('csv/mass_case_description_train_set.csv')

        Y_train = pd.DataFrame()
        Y_test = pd.DataFrame()

        train_labels = pd.concat([Calc_train_labels, Mass_train_labels])
        test_labels = pd.concat([Calc_test_labels, Mass_test_labels])
        class_mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}

        Y_train['id'] = train_labels['patient_id'] + '_' + train_labels['left or right breast'] + '_' + train_labels['image view'] + \
                        '_' + train_labels['abnormality id'].map(str) + '_' + train_labels['abnormality type']
        Y_train['label'] = train_labels['pathology'].map(class_mapping)

        Y_test['id'] = test_labels['patient_id'] + '_' + test_labels['left or right breast'] + '_' + test_labels['image view'] + '_' +\
                       test_labels['abnormality id'].map(str) + '_' + test_labels['abnormality type']
        Y_test['label'] = test_labels['pathology'].map(class_mapping)


        return Y_train, Y_test

def data_to_folders(size=256, data="all", label="diagnosis-4class", show=False, color="GRY"):
    # label = "diagnosis-2class"(benign/malignant) or "diagnosis-4class" or "abnormality"(mass/calc)
    # color = "GRY" or "RGB"

    X_Train_arr = []
    X_Test_arr = []

    Y_Train, Y_Test = np.empty([0]), np.empty([0])
    Y_train, Y_test = organize_labels()

    path = r'D:\Users\Burak\mammograpgy\jpg'
    for dir in os.listdir(path):
        try:
            cropdir = dir[8:]
            if (cropdir[-1].isdigit()):
                print(cropdir)
                abnormality_id = cropdir[-1]
                p_index = cropdir.find('P')
                patient_id = cropdir[p_index + 2: p_index + 7]
                breast = 'LEFT' if 'LEFT' in cropdir else 'RIGHT'
                view = 'MLO' if 'MLO' in cropdir else 'CC'
                abnormality = 'calcification' if cropdir[0:4] == 'Calc' else 'mass'
                is_train = True if 'Training' in cropdir else False
                id = 'P_' + patient_id + '_' + breast + '_' + view + '_' + abnormality_id + '_' + abnormality

                if len(os.listdir(path + '\\' + dir)) == 0:
                    continue
                dir_ = str(os.listdir(path + '\\' + dir)[0])

                if len(os.listdir(path + '\\' + dir + '\\' + dir_)) == 0:
                    continue
                subdir = str(os.listdir(path + '\\' + dir + '\\' + dir_)[0])
                print('     ' + subdir)

                if len(os.listdir(
                        path + '\\' + dir + '\\' + str(os.listdir(path + '\\' + dir)[0]) + '\\' + subdir)) == 0:
                    continue
                subsubdir = str(
                    os.listdir(path + '\\' + dir + '\\' + str(os.listdir(path + '\\' + dir)[0]) + '\\' + subdir)[0])
                print('             ' + subsubdir)

                imagew = cv2.imread(path + '\\' + dir + '\\' + dir_ + '\\' + subdir + '\\' + subsubdir)

                image = clahe(imagew)
                # image = hist_equ(imagew)


                image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

                if(color == "GRY"):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if(show):
                    cv2.imshow(id, image)
                    cv2.waitKey(500)

                if(label == 'diagnosis-2class'):
                    if is_train:
                        if (Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0] == 1):
                            cv2.imwrite("data/diagnosis-2class/" + color + "/train/1/" + str(id) + '.jpg', image)
                        elif(Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0] == 0):
                            cv2.imwrite("data/diagnosis-2class/" + color + "/train/0/" + str(id) + '.jpg', image)
                        else:
                            print("An error occured while saving" + str(id))

                    else:
                        if (Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0] == 1):
                            cv2.imwrite("data/diagnosis-2class/" + color + "/test/1/" + str(id) + '.jpg', image)
                        elif(Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0] == 0):
                            cv2.imwrite("data/diagnosis-2class/" + color + "/test/0/" + str(id) + '.jpg', image)
                        else:
                            print("An error occured while saving" + str(id))
                elif(label == 'diagnosis-4class'):
                    if is_train:
                        if(Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0] == 0):
                            if abnormality == 'calcification':
                                cv2.imwrite("data/diagnosis-4class/" + color + "/train/benign-calc/" + str(id) + '.jpg', image)
                            elif abnormality == 'mass':
                                cv2.imwrite("data/diagnosis-4class/" + color + "/train/benign-mass/" + str(id) + '.jpg', image)
                            else:
                                print("An error occured while saving" + str(id))
                        elif (Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0] == 1):
                            if abnormality == 'calcification':
                                cv2.imwrite("data/diagnosis-4class/" + color + "/train/malignant-calc/" + str(id) + '.jpg', image)
                            elif abnormality == 'mass':
                                cv2.imwrite("data/diagnosis-4class/" + color + "/train/malignant-mass/" + str(id) + '.jpg', image)
                            else:
                                print("An error occured while saving" + str(id))
                        else:
                            print("An error occured while saving" + str(id))

                    else:
                        if(Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0] == 0):
                            if abnormality == 'calcification':
                                cv2.imwrite("data/diagnosis-4class/" + color + "/test/benign-calc/" + str(id) + '.jpg', image)
                            elif abnormality == 'mass':
                                cv2.imwrite("data/diagnosis-4class/" + color + "/test/benign-mass/" + str(id) + '.jpg', image)
                            else:
                                print("An error occured while saving" + str(id))
                        elif (Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0] == 1):
                            if abnormality == 'calcification':
                                cv2.imwrite("data/diagnosis-4class/" + color + "/test/malignant-calc/" + str(id) + '.jpg', image)
                            elif abnormality == 'mass':
                                cv2.imwrite("data/diagnosis-4class/" + color + "/test/malignant-mass/" + str(id) + '.jpg', image)
                            else:
                                print("An error occured while saving" + str(id))
                        else:
                            print("An error occured while saving" + str(id))

                elif(label == 'abnormality'):
                    if is_train:
                        if abnormality == 'calcification':
                            cv2.imwrite("data/abnormality/" + color + "/train/1/" + str(id) + '.jpg', image)
                        else:
                            cv2.imwrite("data/abnormality/" + color + "/train/0/" + str(id) + '.jpg', image)

                    else:
                        if abnormality == 'calcification':
                            cv2.imwrite("data/abnormality/" + color + "/test/1/" + str(id) + '.jpg', image)
                        else:
                            cv2.imwrite("data/abnormality/" + color + "/test/0/" + str(id) + '.jpg', image)

                else:
                    print("*********** INVALID LABEL SET! **********")

        except Exception as err:
            print("***  Exception!  ****")
            print(err)

data_to_folders(size=256, data="all", label="abnormality", show=False, color="RGB")