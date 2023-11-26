import numpy as np
K = np.array([[531.122155322710, 0, 407.192550839899],[0, 531.541737503901, 313.308715048366],[0, 0 ,1]])

def mat(numMatches, cor, img2):
  img2 = float(img2)
  if numMatches == float(2):
    if cor[6] == img2:
      x_mat1 = cor[7]
      y_mat1 = cor[8]
      return np.array([x_mat1, y_mat1])
    else:
      return np.array([-1, -1])
  if numMatches == float(3):
    if cor[6] == img2:
      x_mat1 = cor[7]
      y_mat1 = cor[8]
      return np.array([x_mat1, y_mat1])
    elif cor[9] == img2:
      x_mat1 = cor[10]
      y_mat1 = cor[11]
      return np.array([x_mat1, y_mat1])
    else:
      return np.array([-1, -1])
  if numMatches == float(4):
    if cor[6] == img2:
      x_mat1 = cor[7]
      y_mat1 = cor[8]
      return np.array([x_mat1, y_mat1])
    elif cor[9] == img2:
      x_mat1 = cor[10]
      y_mat1 = cor[11]
      return np.array([x_mat1, y_mat1])
    elif cor[12] == img2:
      x_mat1 = cor[13]
      y_mat1 = cor[14]
      return np.array([x_mat1, y_mat1])
    else:
      return np.array([-1, -1])
  if numMatches == float(5):
    if cor[6] == img2:
      x_mat1 = cor[7]
      y_mat1 = cor[8]
      return np.array([x_mat1, y_mat1])
    elif cor[9] == img2:
      x_mat1 = cor[10]
      y_mat1 = cor[11]
      return np.array([x_mat1, y_mat1])
    elif cor[12] == img2:
      x_mat1 = cor[13]
      y_mat1 = cor[14]
      return np.array([x_mat1, y_mat1])
    elif cor[15] == img2:
      x_mat1 = cor[16]
      y_mat1 = cor[17]
      return np.array([x_mat1, y_mat1])
    else:
      return np.array([-1, -1])
  if numMatches > float(6):
    print('olala')

def find_matches(img1, img2, no_of_images):
    with open('/home/venk/Downloads/Sfm/' + 'matching' + str(img1) + '.txt') as f:
        Prev_contents = f.readlines()
        nfeat = len(Prev_contents) - 1

        contents = []
        for k, line in enumerate(Prev_contents):
            if k != 0:
                line = line.rstrip('\n')
                line_elements = line.split()
                corners = [float(j) for j in line_elements]
                contents.append(corners)

        RGB_vals = []
        coordinates = []
        matchingCoor = []
        feature_flag = np.zeros((1, no_of_images), dtype=int)

        for k, line in enumerate(contents):
            cor = np.array(line)
            numMatches = cor[0] 
            r = cor[1]
            g = cor[2]
            b = cor[3]
            x_cor = cor[4]
            y_cor = cor[5]
            img_id1 = cor[6]
            matching_cor = mat(numMatches, cor, img2)

            RGB_vals.append([r, g, b])
            coordinates.append([x_cor, y_cor])
            matchingCoor.append(matching_cor)

            # Update feature flag for the current image
            feature_flag[0, img1 - 1] = 1

            # Update feature flag for the matched image
            feature_flag[0, img2 - 1] = 1

        # Pad the arrays in matchingCoor with -1 to make them of the same size
        max_size = max(len(arr) for arr in matchingCoor)
        matchingCoor = [np.pad(arr, (0, max_size - len(arr)), constant_values=-1) for arr in matchingCoor]

        matchingCoor = np.array(matchingCoor)
        RGB_vals = np.array(RGB_vals)
        coordinates = np.array(coordinates)
        firstcol = img1 * np.ones((nfeat, 1))
        fourthcol = img2 * np.ones((nfeat, 1))
        InfoAll = np.append(firstcol, RGB_vals, axis=1)
        InfoAll = np.append(InfoAll, coordinates, axis=1)
        InfoAll = np.append(InfoAll, fourthcol, axis=1)
        InfoAll = np.append(InfoAll, matchingCoor, axis=1)

        count = 0
        for k, j in enumerate((matchingCoor)):
            if j[0] == -1:
                count = count + 1

        newinfo = []
        for row in InfoAll:
            if row[7] != -1:
                newinfo.append(row)

        newinfo = np.array(newinfo)
        return newinfo

def features_extraction(data):

    no_of_images = 5
    feature_rgb_values = []
    feature_x = []
    feature_y = []

    "We have 4 matching.txt files"
    feature_flag = []

    for n in range(1, no_of_images):
        file = data + "/matching" + str(n) + ".txt"
        matching_file = open(file,"r")
        nfeatures = 0

        for i, row in enumerate(matching_file):
            if i == 0:  #1st row having nFeatures/no of features
                row_elements = row.split(':')
                nfeatures = int(row_elements[1])
            else:
                x_row = np.zeros((1,no_of_images))
                y_row = np.zeros((1,no_of_images))
                flag_row = np.zeros((1,no_of_images), dtype = int)
                row_elements = row.split()
                columns = [float(x) for x in row_elements]
                columns = np.asarray(columns)

                nMatches = columns[0]
                r_value = columns[1]
                b_value = columns[2]
                g_value = columns[3]

                feature_rgb_values.append([r_value,g_value,b_value])
                current_x = columns[4]
                current_y = columns[5]

                x_row[0,n-1] = current_x
                y_row[0,n-1] = current_y
                flag_row[0,n-1] = 1

                m = 1
                while nMatches > 1:
                    image_id = int(columns[5+m])
                    image_id_x = int(columns[6+m])
                    image_id_y = int(columns[7+m])
                    m = m+3
                    nMatches = nMatches - 1

                    x_row[0, image_id - 1] = image_id_x
                    y_row[0, image_id - 1] = image_id_y
                    flag_row[0, image_id - 1] = 1

                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)

    feature_x = np.asarray(feature_x).reshape(-1,no_of_images)
    feature_y = np.asarray(feature_y).reshape(-1,no_of_images)
    feature_flag = np.asarray(feature_flag).reshape(-1,no_of_images)
    feature_rgb_values = np.asarray(feature_rgb_values).reshape(-1,3)

    return feature_x, feature_y, feature_flag, feature_rgb_values