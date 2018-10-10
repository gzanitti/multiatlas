import numpy as np

def label_votes(train_labels):
    """Returns a matrix M of size subjects x labels x voxels.
       M[n, l, x] = 1 if the subject n labeled the voxel x as l
       M[n, l, x] = 0 if not
       
       Summing trought the coordinate 0, we can obtain the amount
       of votes a label obtained in a specific voxel"""
    train_labels = np.atleast_2d(train_labels)
    
    if len(train_labels.shape) > 2:
        raise ValueError("Please input a matrix of size 'nsubjects' x 'nvoxels'")
        
    labels = np.unique(train_labels)  
    voting = np.array([[tl == l for l in labels] for tl in train_labels])
    return voting


def intensity_id(test_image, train_images):
    test_image = np.atleast_1d(test_image)
    train_images = np.atleast_2d(train_images)
    
    if len(train_images.shape) > 2:
        raise ValueError("Please input a 'train_images' matrix of size 'nsubjects' x 'nvoxels'")

    if len(test_image.shape) > 1:
        raise ValueError("Please input a 'test_image' matrix of size 'nvoxels'")
    
    if test_image.shape[0] != train_images.shape[1]:
        raise ValueError("The 'test_image' and the 'train_images' should contain the same number of voxels (dim 1)")
    
    return np.ones_like(train_images)


def diffusion_id(test_diffusion, train_diffusion, train_labels):
    test_diffusion = np.atleast_2d(test_diffusion)
    train_diffusion = np.atleast_2d(train_diffusion)
    train_labels = np.atleast_2d(train_labels)
    
    if len(train_labels.shape) > 2:
        raise ValueError("Please input a 'train_labels' matrix of size 'nvoxels' x 'nsubjects'")
    
    if train_diffusion.shape[0] != train_labels.shape[0]:
        raise ValueError("All the first dimentions should have the same size, representing an equal amount of subjects")        

    if test_diffusion.shape[0] != train_diffusion.shape[1] or test_diffusion.shape[0] != train_labels.shape[1]:
        raise ValueError("All the matrices should have the same number of voxels")
        
    labels = np.unique(train_labels)
    
    return np.ones((train_labels.shape[0], len(labels), train_labels.shape[1]))
