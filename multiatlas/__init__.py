import numpy as np

def generative_model(probabilities_labels, intensity_similarity, diffusion_similarity,
                     train_labels, test_image, train_images, test_diffusion, train_diffusion):
    """Implementation of the Generative model
        
       Parameters
       ----------
       probabilities_labels: function
           Function that takes as input the train_labels. The output of this function
           MUST be a matrix pL of size subjects x labels x voxels, representing
           pL[n,l,x] = P(L[x]=l|Ln)
       intensity_similarity: function
           Function that takes as input the test_image and the train_images, in that
           order. The output of this function MUST be a matrix pI of size 
           subjects x voxels, representing pI[n, x] = P(I[x]|In)
       diffusion_similarity: function
           Function that takes as input the test_diffusion, train_diffusion and
           train_labels, in that order. The output of this function MUST be a matrix pD
           of size subjects x label x voxels, representing pD[n, l, x] = P(D[x]|Dn[x,l])
       train_labels: object
           The labeling of the train subjects. This object will be fed into the
           functions probabilities_labels and diffusion_similarity
       test_image: object
           The tissue-intensities of each voxel. This object will be fed into the
           function intensity_similarity
       train_images: object
           The tissue-intensities of the voxels of each train subject. This object
           will be fed into the function intensity_similarity
       test_diffusion: object
           The diffusion information of the test subject. This object will be fed
           into the function diffusion_similarity
       train_diffusion: object
           The diffusion information of each train subject. This object will be fed
           into the function diffusion_similarity
       
       Returns
       -------
       labeling: array-like
           An array of size 'number of voxels', with the fused labeling.
       """
    # pL = Matrix with size subjects x labels x voxels. pL[n,l,x] = P(L[x]=l|Ln)
    pL = probabilities_labels(train_labels)

    # pI = Matrix with size subjects x voxels. pI[n, x] = P(I[x]|In)
    pI = intensity_similarity(test_image, train_images)

    # pD = Matrix with size subjects x label x voxels. pD[n, l, x] = P(D[x]|Dn[x,l])
    pD = diffusion_similarity(test_diffusion, train_diffusion, train_labels)

    # Reshape matrix pI and tile it (respecting the number of labels) to multiply it
    pIr = pI.reshape((pI.shape[0], 1, pI.shape[1]))
    pIr = np.tile(pIr, (1, pL.shape[1], 1))

    # Multiply the probabilities/similarities and sum across subjects
    sum_over_subjects = (pL*pIr*pD).sum(0)

    # Retrieve the labels with maximum probability
    return np.argmax(sum_over_subjects, axis=0)
