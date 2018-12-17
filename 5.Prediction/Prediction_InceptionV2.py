'''
All rights reserved", Royan Institute for Stem Cell Biology and Technology,
Oct 2017  ( Mehdi Habibzadeh et al)
'''



from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
# import inception
import inception_preprocessing
from nets import inception
# from preprocessing import inception_preprocessing

slim = tf.contrib.slim

#checkpoints path of fine-tunning last layer
checkpoints_path = '/home/habibzadeh/Desktop/BC_malignant_checkpoints/inception_v2'

#checkpoints path of fine-tunning all layers
#checkpoints_path = '/home/habibzadeh/Desktop/BC_malignant_checkpoints/inception_v2/all/'
checkpoints_path = tf.train.latest_checkpoint(checkpoints_path)
print("checkpoint path is",checkpoints_path)

image_size = inception.inception_v2.default_image_size

with tf.Graph().as_default():
    image_string=tf.gfile.GFile("./Malignant_Test/Mucinous.JPEG", "rb").read()

    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)

    processed_images = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception.inception_v2_arg_scope()):
        logits, _ = inception.inception_v2(processed_images, num_classes=4, is_training=False)

    probabilities = tf.nn.softmax(logits)


    init_fn = slim.assign_from_checkpoint_fn(
        checkpoints_path,
        slim.get_model_variables('InceptionV2'))

    with tf.Session() as sess:
        init_fn(sess)

        # We want to get predictions, image as numpy matrix
        # and resized and cropped piece that is actually
        # being fed to the network.
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x: x[1])]


    # Show the downloaded image
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Input image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    # Show the image that is actually being fed to the network
    # The image was resized while preserving aspect ratio and then
    # cropped. After that, the mean pixel value was subtracted from
    # each pixel of that crop. We normalize the image to be between [-1, 1]
    # to show the image.
    plt.imshow(network_input / (network_input.max() - network_input.min()))
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    names = ["Ductal carcinoma","Lobular carcinoma","Mucinous carcinoma","Papillary carcinoma"]
    for i in range(4):
        index = sorted_inds[i]
        print(index)

        # Now we print the top-5 predictions that the network gives us with
        # corresponding probabilities.
        print('Probability %0.4f => [%s]' % (probabilities[index], names[index]))



