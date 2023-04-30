import numpy as np


def ErrorComparison(K,R2,C2,bestWP,Xbest,inliers1and2):

    linear_error=0
    non_linear_error=0

    


    temp2 = np.hstack((np.identity((3)),-C2))
    PM2 = np.matmul(K, np.matmul(R2,temp2))


    for pt_linear,pt_non,pt2_in in zip(bestWP,Xbest,inliers1and2[:,7:9]):

        # print(pt1_in)

        pt1=np.append(pt_linear,[1])
        pt2=np.append(pt_non,[1])

        ###### LINEAR ##################
        #### Project-it-in-image ####



        proj_2_x=np.matmul(PM2[0,:].T,pt1)
        proj_2_y=np.matmul(PM2[1,:].T,pt1)
        proj_2_z=np.matmul(PM2[2,:].T,pt1)

        proj_2_x/=proj_2_z
        proj_2_y/=proj_2_z

        proj_2_x=abs(proj_2_x)
        proj_2_y=abs(proj_2_y)



        #### Get Error ######

        linear_error+=np.sqrt((proj_2_y-pt2_in[0])**2 + (proj_2_x-pt2_in[1])**2)


        #### NON LINEAR TRIANGULATION #############

        proj_2_x=np.matmul(PM2[0,:].T,pt2)
        proj_2_y=np.matmul(PM2[1,:].T,pt2)
        proj_2_z=np.matmul(PM2[2,:].T,pt2)

        proj_2_x/=proj_2_z
        proj_2_y/=proj_2_z

        proj_2_x=abs(proj_2_x)
        proj_2_y=abs(proj_2_y)


        non_linear_error+=np.sqrt((proj_2_y-pt2_in[0])**2 + (proj_2_x-pt2_in[1])**2)

    print('Linear Error is:',linear_error)
    print('Non Linear Error is:',non_linear_error)





