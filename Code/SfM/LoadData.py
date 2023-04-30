import numpy as np



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

def findMatches(img1, img2):
  with open('../Data/P3Data/'+ 'matching' + str(img1) + '.txt') as f:
      Prev_contents = f.readlines()       
      nfeat = len(Prev_contents) - 1
      # print(f"number of features in matching{img1}.txt are {nfeat}")
      # print(Prev_contents)
      contents = []
      for k,line in enumerate(Prev_contents):
        if k != 0:
          # contents = [float(i) for i in line]
          line = line.rstrip('\n')
          line_elements = line.split()
          corners = [float(j) for j in line_elements]
          # print(type(cor[random.randint(1,6)]))
          contents.append(corners)
      # print(contents)
      RGB_vals = []
      coordinates = []
      matchingCoor = []
      for k,line in enumerate(contents):
        # if k == 963:  
          cor = np.array(line)
          numMatches = cor[0]
          r = cor[1]
          g = cor[2]
          b = cor[3]
          x_cor = cor[4]
          y_cor = cor[5]
          img_id1 = cor[6]
          matching_cor = mat(numMatches, cor, img2)
          # print(type(matching_cor))
          # print(matching_cor)
          RGB_vals.append([r,g,b])
          coordinates.append([x_cor, y_cor])
          # if matching_cor != None:
          matchingCoor.append(matching_cor)
      matchingCoor = np.array(matchingCoor)
      # matchingCoor[matchingCoor==-1]=[-1,-1]
      # print(matchingCoor)
      matchingCoor = np.reshape(matchingCoor, (nfeat,2))
      RGB_vals = np.array(RGB_vals)
      coordinates = np.array(coordinates)
      firstcol = img1*np.ones((nfeat,1))
      fourthcol = img2*np.ones((nfeat,1))
      # print(f"firstcol={firstcol.shape}, RGB_vals={RGB_vals.shape}, coordinates={coordinates.shape},fourthcol={fourthcol.shape}, matchingCoor={matchingCoor.shape}")
      InfoAll = np.append(firstcol, RGB_vals,  axis = 1)
      InfoAll = np.append(InfoAll, coordinates,  axis = 1)
      InfoAll = np.append(InfoAll, fourthcol,  axis = 1)
      InfoAll = np.append(InfoAll, matchingCoor,  axis = 1)
      # print(InfoAll.shape)
      # print((matchingCoor.shape[0]))
      count = 0
      for k,j in enumerate((matchingCoor)):
        if j[0] == -1:
          count = count+1
      # print(count)
      # print(InfoAll)
      newinfo = []
      for row in InfoAll:
        if row[7] != -1:
          newinfo.append(row)
      newinfo = np.array(newinfo) 
      # print(newinfo.shape)
      return newinfo


